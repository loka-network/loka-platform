"""Workflow A — ontology generation from domain texts.

    domain texts + prompt --LLM--> ontology + acquired knowledge (DATA / METHODS) --> KB

Same discipline as grounding: a builder *proposes* a draft; the type system *disposes* —
:func:`build` compiles the draft into an ontology definition and runs it through
``load_ontology_str``, so a malformed or inconsistent proposal is rejected, not trusted.

Two proposers ship: a deterministic ``KeywordBuilder`` (no LLM — for tests / offline / sovereign
runs) that extracts entity types, subtype-free relations, and action verbs from the text; and an
opt-in ``LLMBuilder`` that additionally proposes subtypes and typed attributes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loka_schemas import KBSpec

from .loader import load_ontology_str

# Sentence-initial / common capitalized words that are not entity types.
_STOP = frozenset(
    {
        "The", "A", "An", "If", "When", "This", "That", "It", "We", "They", "Then",
        "For", "In", "On", "Of", "To", "And", "But", "Or", "As", "At", "By", "So",
        "Analysts", "Given", "Which", "Change",
    }
)
_ENT = r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*"
_TITLE = re.compile(rf"\b({_ENT})\b")
_REL_VERBS = (
    "sets", "affects", "influences", "causes", "drives", "determines",
    "moves", "raises", "lowers", "impacts",
)
_REL_RE = re.compile(rf"({_ENT})\s+(" + "|".join(_REL_VERBS) + rf")\s+(?:the\s+)?({_ENT})")
_METHOD_WORDS = {
    "forecast": "forecast", "predict": "forecast",
    "compare": "rank", "rank": "rank",
    "effect": "causal_effect", "impact": "causal_effect", "cause": "causal_effect",
}
_BASE_TYPE = {
    "string": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "double", "double": "double", "number": "double",
    "bool": "boolean", "boolean": "boolean",
    "timestamp": "timestamp", "datetime": "timestamp", "date": "date",
}
_VERB_CLASS = {"factual", "communicative", "institutional"}

#: Output budget every extraction asks for, whichever paradigm it belongs to.
#:
#: One number, because it is not only a ceiling — it changes how much the model writes. Measured
#: on the same document with the same prompt, deepseek-chat returned 874 characters at 4,096 and
#: 2,819 at 16,000, both with finish_reason=stop: neither was truncated, it simply wrote more
#: when allowed more. So paradigms given different budgets are not comparable, and the
#: single-shot route asking for 4,000 while carrying an entire ontology in one reply was the
#: most starved of them.
EXTRACTION_MAX_TOKENS = 16000

#: Sampling temperature for every extraction.
#:
#: Zero, because nothing here wants variety. Left unset, the endpoint samples at its own default
#: and every run draws again: the same paradigm over the same document with the same prompt
#: returned forty entity types once and nine the next time. Every comparison run before this was
#: therefore a single sample of a wide distribution being read as a measurement, and the
#: conclusions drawn from them do not hold. It also matters beyond experiments — a customer who
#: reviews and publishes an ontology should be able to rebuild it and get the same one.
EXTRACTION_TEMPERATURE = 0.0


class OntologyBuildError(RuntimeError):
    """The model replied, but not with an ontology this can read. Carries what came back: a
    decision about a prompt or a token budget cannot be made from a character offset."""


@dataclass(frozen=True)
class EntityDraft:
    """A proposed entity type: name + optional supertype + typed attributes."""

    name: str
    subtype_of: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()  # (attr_name, base_type)


@dataclass(frozen=True)
class OntologyDraft:
    """An unvalidated proposal from a builder (before the type system disposes)."""

    entities: tuple[EntityDraft, ...]
    relations: tuple[tuple[str, str, str], ...] = ()  # (from, verb, to)
    verbs: tuple[tuple[str, str], ...] = ()  # (name, class)
    data_needs: tuple[str, ...] = ()
    method_needs: tuple[str, ...] = ()
    facets: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entities)


@runtime_checkable
class OntologyBuilder(Protocol):
    """Turns domain texts into an unvalidated :class:`OntologyDraft`."""

    def propose(self, texts: Sequence[str]) -> OntologyDraft: ...


def _camel(phrase: str) -> str:
    words = phrase.split()
    while words and words[0] in _STOP:  # drop leading "The"/"A"/... from the phrase
        words.pop(0)
    return "".join(w[:1].upper() + w[1:] for w in words)


class KeywordBuilder:
    """Deterministic, no-LLM reference builder.

    Entity types = distinct Title-Case noun phrases (CamelCased). Relations = "<E> <verb> <E>"
    patterns; those verbs also become action verbs. Method needs = detected task words. No
    subtypes/attributes (a keyword pass cannot reliably infer them — the LLM builder does).
    """

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        blob = "\n".join(texts)
        names: list[str] = []
        for phrase in _TITLE.findall(blob):
            name = _camel(phrase)
            if name and name not in names:
                names.append(name)
        known = set(names)

        relations: list[tuple[str, str, str]] = []
        verbs: dict[str, str] = {}
        for left, verb, right in _REL_RE.findall(blob):
            src, tgt = _camel(left), _camel(right)
            if src in known and tgt in known and (src, verb, tgt) not in relations:
                relations.append((src, verb, tgt))
                verbs[verb.upper()] = "factual"

        methods: list[str] = []
        low = blob.lower()
        for word, method in _METHOD_WORDS.items():
            if word in low and method not in methods:
                methods.append(method)

        return OntologyDraft(
            entities=tuple(EntityDraft(name=n) for n in names),
            relations=tuple(relations),
            verbs=tuple(verbs.items()),
            data_needs=tuple(names),
            method_needs=tuple(methods),
            facets={"factual": tuple(names), "cognitive": tuple(methods), "communication": ()},
        )


class LLMBuilder:
    """Opt-in model-backed builder. Injectable client (any object with ``messages.create``)."""

    _SYSTEM = (
        "Extract an ontology from the domain text. Reply with ONLY a JSON object: "
        '{"entities": [{"name": <CamelCase>, "subtype_of": <name|null>, '
        '"attributes": [{"name": <str>, "type": "string|integer|double|boolean|timestamp"}]}], '
        '"relations": [[from, verb, to], ...], '
        '"verbs": [[NAME, "factual|communicative|institutional"]], '
        '"data_needs": [...], "method_needs": [...]}. No prose, no code fences.'
    )

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "claude-opus-4-8",
        system_prompt: str | None = None,
    ) -> None:
        """``system_prompt`` overrides the default extraction instruction.

        The prompt is the method: it decides which parts of Ω can come out of a text at all, so
        two runs over one document with different prompts are two different procedures and not
        two samples of one. It is a parameter so a domain can supply its own, and it is readable
        afterwards (:attr:`system_prompt`) so what ran can be recorded rather than reconstructed
        from a copy kept somewhere else, which is the kind of copy that goes stale.
        """
        if client is None:
            import anthropic  # optional [llm] extra, imported lazily

            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._system = system_prompt or self._SYSTEM

    @property
    def system_prompt(self) -> str:
        """The instruction this builder sends, verbatim."""
        return self._system

    @property
    def model(self) -> str:
        return self._model

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
            system=self._system,
            messages=[{"role": "user", "content": "\n".join(texts)}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        start, end = text.find("{"), text.rfind("}")
        if start == -1:
            raise OntologyBuildError(
                f"the model returned no JSON object ({len(text)} chars): {text[:200]!r}"
            )
        # A reply cut off before any closing brace has no ``end``, and slicing to it would give
        # the empty string — an error report showing nothing, in the one case where seeing the
        # text is the whole point. Take the tail instead.
        candidate = text[start : end + 1] if end > start else text[start:]
        try:
            obj: dict[str, Any] = json.loads(candidate)
        except json.JSONDecodeError as exc:
            # A cut-off reply and a malformed one both arrive here, and a bare decoder message
            # names a character position in text the operator never sees — which reads as "the
            # model writes bad JSON" when the usual cause is that it was not allowed to finish.
            # The two are told apart by whether the braces balance, and the text is shown either
            # way, because a decision about a prompt cannot be made from an offset.
            truncated = candidate.count("{") > candidate.count("}")
            raise OntologyBuildError(
                ("the model's reply was cut off mid-JSON" if truncated
                 else "the model returned malformed JSON")
                + f" at char {exc.pos} of {len(candidate)} ({exc.msg}). "
                + f"It ends: ...{candidate[-200:]!r}"
            ) from exc

        entities: list[EntityDraft] = []
        for e in obj.get("entities", []):
            if not isinstance(e, dict) or not e.get("name"):
                continue
            attrs = tuple(
                (a["name"], _BASE_TYPE.get(str(a.get("type", "string")).lower(), "string"))
                for a in e.get("attributes", [])
                if isinstance(a, dict) and a.get("name")
            )
            entities.append(
                EntityDraft(
                    name=e["name"], subtype_of=e.get("subtype_of") or None, attributes=attrs
                )
            )
        rels = tuple(
            (r[0], r[1], r[2])
            for r in obj.get("relations", [])
            if isinstance(r, list) and len(r) == 3
        )
        verbs = tuple(
            (v[0], v[1] if v[1] in _VERB_CLASS else "factual")
            for v in obj.get("verbs", [])
            if isinstance(v, list) and len(v) == 2
        )
        names = tuple(e.name for e in entities)
        return OntologyDraft(
            entities=tuple(entities),
            relations=rels,
            verbs=verbs,
            data_needs=tuple(obj.get("data_needs", [])) or names,
            method_needs=tuple(obj.get("method_needs", [])),
            facets={"factual": names, "cognitive": tuple(obj.get("method_needs", []))},
        )


def analyze_facets(draft: OntologyDraft) -> dict[str, tuple[str, ...]]:
    """Decompose the draft into three facets.

    The three faces of a cognitive agent — factual, cognitive, communicative:

      * **Factual** — the objective/interobjective world: entity types, their typed attributes,
        relations, and factual verbs. This is the "what is" that KB.DATA stores.
      * **Cognitive** — the reasoning/decision content: the methods (competencies) the agent can
        apply, i.e. KB.METHODS. (Epistemic/deontic/intentional acts live here as they are added.)
      * **Communication** — the communication acts the agent performs: ``informs / asks / orders``
        (the speech acts realized in ``speechact.py``), plus any communicative verbs; institutional
        verbs (declares/compels/authorizes) are surfaced here too.
    """
    factual: list[str] = []
    for e in draft.entities:
        factual.append(e.name)
        factual.extend(f"{e.name}.{aname}" for aname, _ in e.attributes)
    factual.extend(f"{s} -{v}-> {t}" for s, v, t in draft.relations)
    factual.extend(f"verb:{n}" for n, c in draft.verbs if c == "factual")

    cognitive = [f"method:{m}" for m in draft.method_needs]

    communication = ["informs", "asks", "orders"]  # the acts speechact.py realizes
    communication.extend(f"verb:{n}" for n, c in draft.verbs if c == "communicative")
    communication.extend(f"institutional:{n}" for n, c in draft.verbs if c == "institutional")

    return {
        "factual": tuple(dict.fromkeys(factual)),
        "cognitive": tuple(dict.fromkeys(cognitive)),
        "communication": tuple(dict.fromkeys(communication)),
    }


def build(texts: Sequence[str], builder: OntologyBuilder | None = None) -> KBSpec:
    """Workflow A end-to-end: propose a draft, then compile + validate it into a KBSpec.

    Raises ``OntologyLoadError`` if the proposed ontology is inconsistent (the type system
    disposing of a bad proposal) — the caller sees a structured failure, never a silent bad KB.
    The KBSpec carries the three-facet analysis in ``facets`` (factual/cognitive/communication).
    """
    from .verb_syntax import to_verb_syntax

    builder = builder or KeywordBuilder()
    draft = builder.propose(texts)
    # Parsed before it is handed on, which is the validation step, and re-emitted from the
    # parsed object rather than from the draft: what the reviewer reads is then the ontology
    # that loaded, not a second rendering of it that could drift from what was checked.
    ontology = load_ontology_str(_draft_to_yaml(draft))
    yaml = to_verb_syntax(ontology)
    return KBSpec(
        ontology_yaml=yaml,
        data_needs=draft.data_needs,
        method_needs=draft.method_needs,
        facets=analyze_facets(draft),  # the three-facet analysis
    )


def _draft_to_yaml(draft: OntologyDraft) -> str:
    known = set(draft.entity_names)
    lines = ["version: built-v0.1", "entities:"]
    for e in draft.entities:
        lines.append(f"  - type: {e.name}")
        if e.subtype_of and e.subtype_of in known and e.subtype_of != e.name:
            lines.append(f"    subtype_of: {e.subtype_of}")
        if e.attributes:
            lines.append("    properties:")
            for aname, atype in e.attributes:
                safe = re.sub(r"[^A-Za-z0-9_]", "_", aname)
                lines.append(f"      - {{name: {safe}, type: {atype}}}")
    verbs = [(re.sub(r"[^A-Za-z0-9_]", "_", n).upper(), c) for n, c in draft.verbs]
    if verbs:
        lines.append("verbs:")
        for name, cls in verbs:
            lines.append(f"  - {{name: {name}, class: {cls}}}")
    rels = [(s, v, t) for s, v, t in draft.relations if s in known and t in known]
    if rels:
        lines.append("relations:")
        for src, verb, tgt in rels:
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", verb) or "relates"
            lines.append(f"  - {{name: {safe}, from: {src}, to: {tgt}}}")
    return "\n".join(lines) + "\n"
