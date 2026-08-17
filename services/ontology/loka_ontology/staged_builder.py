"""Extract an ontology in stages, and make every proposal cite the text it came from.

The single-shot builder asks one question — "read this document and return the whole ontology" —
and takes whatever comes back. Two things go wrong with that, and we hit both on the first real
document.

The reply has to carry the entire ontology at once, so on a domain of any size it is cut off by
the token budget. Splitting the *document* does not fix this: it is the *answer* that is too
long. Splitting the *task* does, because each stage's answer is small no matter how large the
domain. That is the argument in the LLM4Onto paper (Ouyang et al.), which names the single-shot
form Text→Ontology and treats it as the baseline to improve on; this module implements the
decomposition side of that argument.

And nothing checks that what came back is about the document. A model asked for an ontology will
produce a plausible one — for marketplaces in general, if the text underdetermines it. The paper
prevents this by construction: it clusters noun phrases actually present in the text and asks the
model only to *name* the clusters, so no concept can be invented. That needs a POS tagger, an
embedding model and a clustering pass.

This takes the same principle from the other end, which fits a system whose whole posture is that
model output is not trusted but checked: every stage must return, alongside each proposal, the
span of source text it read it from — and the span is then looked for in the source. A concept
the document does not mention is not silently dropped, because it may be a correct generalisation
("Shopper" for a document that says "customer"); it is marked ungrounded and put in front of the
reviewer. Detection rather than prevention, with the disagreement made visible either way.

Four stages, each a separate call with a small answer:

    1  entity types          + the phrase in the text denoting each
    2  attributes per type   + the phrase, in batches so the answer stays short
    3  relations             + the sentence stating each, as a triple over stage-1 types
    4  verbs and act classes over the relations found

Stage 3 is the one the paper's HT-R-O paradigm is about: the model is not asked what relation
*ought* to hold between two concepts, but what a given sentence *says* about them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .builder import _BASE_TYPE, _VERB_CLASS, EntityDraft, OntologyBuildError, OntologyDraft

#: Entities per call in stage 2. Small enough that a batch's answer cannot exhaust the budget,
#: large enough that a twenty-type ontology does not become twenty round trips.
_ATTR_BATCH = 6

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Grounding:
    """Which proposals were found in the source text and which were not.

    Kept as a record rather than applied as a filter: a name absent from the text is not thereby
    wrong. "Shopper" for a document that says "customer", "PurchaseLine" for one that says "each
    line of a purchase" — both are correct generalisations no string match will confirm. What is
    not acceptable is the two being indistinguishable in the result.
    """

    grounded: dict[str, str] = field(default_factory=dict)  # proposal -> evidence found
    ungrounded: dict[str, str] = field(default_factory=dict)  # proposal -> evidence claimed

    def note(self, proposal: str, evidence: str, source: str) -> bool:
        ok = _appears_in(evidence, source) or _appears_in(proposal, source)
        (self.grounded if ok else self.ungrounded)[proposal] = evidence
        return ok

    def as_dict(self) -> dict[str, Any]:
        total = len(self.grounded) + len(self.ungrounded)
        return {
            "checked": total,
            "grounded": len(self.grounded),
            "ungrounded": sorted(self.ungrounded),
            "rate": round(len(self.grounded) / total, 4) if total else 0.0,
            "evidence": dict(self.grounded),
        }


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _appears_in(phrase: str, source: str) -> bool:
    """Whether a phrase occurs in the source, allowing for the ways a name is written.

    Compared on word tokens rather than characters, so that CamelCase, snake_case and spacing
    do not decide the answer: "PurchaseLine" is looked for as "purchase line", and a trailing
    plural is tolerated. Deliberately not fuzzy beyond that — a check that matches anything
    reports everything as grounded and is worth nothing.
    """
    if not phrase:
        return False
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", phrase)
    want = _tokens(spaced)
    if not want:
        return False
    have = _tokens(source)

    def same(a: str, b: str) -> bool:
        return a == b or a.rstrip("s") == b.rstrip("s")

    n = len(want)
    return any(
        all(same(want[j], have[i + j]) for j in range(n))
        for i in range(len(have) - n + 1)
    )


def _json_object(text: str, stage: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise OntologyBuildError(f"stage '{stage}' returned no JSON object: {text[:200]!r}")
    candidate = text[start : end + 1] if end > start else text[start:]
    try:
        obj: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        cut = candidate.count("{") > candidate.count("}")
        raise OntologyBuildError(
            (f"stage '{stage}' was cut off mid-JSON" if cut else f"stage '{stage}' returned "
             "malformed JSON")
            + f" at char {exc.pos} of {len(candidate)} ({exc.msg}). Ends: ...{candidate[-160:]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise OntologyBuildError(f"stage '{stage}' returned {type(obj).__name__}, not an object")
    return obj


class StagedLLMBuilder:
    """Four small extractions instead of one large one, each citing the text."""

    ENTITIES = (
        "List the entity types a domain ontology for this text needs. An entity type is a kind "
        "of thing the domain has many of, not a one-off. Reply with ONLY JSON: "
        '{"entities": [{"name": <CamelCase>, "subtype_of": <name|null>, '
        '"evidence": <the exact phrase from the text that denotes it>}]}. '
        "The evidence must be copied from the text, not paraphrased. If a type is implied rather "
        "than named, give the phrase that implies it. No prose, no code fences."
    )
    ATTRIBUTES = (
        "For each entity type listed, give the attributes the text states it has. Reply with "
        'ONLY JSON: {"attributes": {"<EntityType>": [{"name": <snake_case>, '
        '"type": "string|integer|double|boolean|timestamp|date", '
        '"evidence": <the exact phrase from the text>}]}}. '
        "Only attributes the text actually mentions. An empty list is a valid answer. "
        "No prose, no code fences."
    )
    RELATIONS = (
        "Given these entity types, extract the relations the text states between them. Do not "
        "propose relations that would be reasonable in this domain — only ones this text asserts. "
        'Reply with ONLY JSON: {"relations": [{"from": <EntityType>, "verb": <lower_snake>, '
        '"to": <EntityType>, "evidence": <the sentence from the text that states it>}]}. '
        "The evidence must be a sentence copied from the text. No prose, no code fences."
    )
    VERBS = (
        "Classify each relation verb by act class. factual = a change or fact in the objective "
        "world; communicative = a message between parties; institutional = an act that changes "
        'standing or authority by declaration. Reply with ONLY JSON: {"verbs": [[<VERB>, '
        '"factual|communicative|institutional"]]}. No prose, no code fences.'
    )

    def __init__(
        self, *, client: Any, model: str = "claude-opus-4-8", max_tokens: int = 4000
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self.grounding = Grounding()
        self.stage_calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def system_prompt(self) -> str:
        """All four instructions, in order. What ran, for the record."""
        return "\n\n---\n\n".join(
            f"[stage {i}] {p}"
            for i, p in enumerate((self.ENTITIES, self.ATTRIBUTES, self.RELATIONS, self.VERBS), 1)
        )

    def _ask(self, stage: str, system: str, user: str) -> dict[str, Any]:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        obj = _json_object(text, stage)
        self.stage_calls.append({"stage": stage, "reply_chars": len(text)})
        return obj

    # ---- stages ----

    def _entities(self, source: str) -> list[EntityDraft]:
        obj = self._ask("entities", self.ENTITIES, source)
        out: list[EntityDraft] = []
        seen: set[str] = set()
        for e in obj.get("entities", []):
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            self.grounding.note(name, str(e.get("evidence") or ""), source)
            out.append(EntityDraft(name=name, subtype_of=e.get("subtype_of") or None))
        # A subtype pointing at a type that was not proposed cannot load, and dropping the link
        # keeps the type: losing "BulkyItem is a kind of Item" is better than losing BulkyItem.
        names = {e.name for e in out}
        return [
            EntityDraft(name=e.name, subtype_of=e.subtype_of if e.subtype_of in names else None)
            for e in out
        ]

    def _attributes(
        self, source: str, entities: list[EntityDraft]
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        found: dict[str, tuple[tuple[str, str], ...]] = {}
        for i in range(0, len(entities), _ATTR_BATCH):
            batch = entities[i : i + _ATTR_BATCH]
            names = ", ".join(e.name for e in batch)
            obj = self._ask(
                f"attributes[{i // _ATTR_BATCH}]",
                self.ATTRIBUTES,
                f"Entity types: {names}\n\nText:\n{source}",
            )
            for entity, attrs in (obj.get("attributes") or {}).items():
                if entity not in {e.name for e in batch} or not isinstance(attrs, list):
                    continue
                collected: list[tuple[str, str]] = []
                seen: set[str] = set()
                for a in attrs:
                    if not isinstance(a, dict) or not a.get("name"):
                        continue
                    aname = re.sub(r"[^A-Za-z0-9_]", "_", str(a["name"])).strip("_")
                    if not aname or aname in seen:
                        continue
                    seen.add(aname)
                    self.grounding.note(
                        f"{entity}.{aname}", str(a.get("evidence") or ""), source
                    )
                    collected.append(
                        (aname, _BASE_TYPE.get(str(a.get("type", "string")).lower(), "string"))
                    )
                found[entity] = tuple(collected)
        return found

    def _relations(
        self, source: str, entities: list[EntityDraft]
    ) -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
        names = {e.name for e in entities}
        obj = self._ask(
            "relations",
            self.RELATIONS,
            f"Entity types: {', '.join(sorted(names))}\n\nText:\n{source}",
        )
        rels: list[tuple[str, str, str]] = []
        evidence: dict[str, str] = {}
        for r in obj.get("relations", []):
            if not isinstance(r, dict):
                continue
            src, verb, tgt = (
                str(r.get("from") or ""), str(r.get("verb") or ""), str(r.get("to") or "")
            )
            # A relation between types that were never proposed cannot load. Silently dropping it
            # would hide a disagreement between two stages, so it is recorded as ungrounded.
            if src not in names or tgt not in names or not verb:
                self.grounding.ungrounded[f"{src} -{verb}-> {tgt}"] = "endpoint not an entity type"
                continue
            key = f"{src} -{verb}-> {tgt}"
            if key in evidence:
                continue
            self.grounding.note(key, str(r.get("evidence") or ""), source)
            evidence[key] = str(r.get("evidence") or "")
            rels.append((src, verb, tgt))
        return tuple(rels), evidence

    def _verbs(self, relations: Sequence[tuple[str, str, str]]) -> tuple[tuple[str, str], ...]:
        verbs = sorted({v for _, v, _ in relations})
        if not verbs:
            return ()
        obj = self._ask("verbs", self.VERBS, "Verbs: " + ", ".join(verbs))
        classified = {
            str(v[0]).upper(): (str(v[1]) if str(v[1]) in _VERB_CLASS else "factual")
            for v in obj.get("verbs", [])
            if isinstance(v, list) and len(v) == 2
        }
        # A verb the classifier skipped still has to exist, or the relation using it will not
        # load. Defaulting to factual is a decision the review checklist can surface.
        return tuple((v.upper(), classified.get(v.upper(), "factual")) for v in verbs)

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        source = "\n".join(texts)
        entities = self._entities(source)
        if not entities:
            raise OntologyBuildError("stage 'entities' proposed no entity types")
        attrs = self._attributes(source, entities)
        entities = [
            EntityDraft(name=e.name, subtype_of=e.subtype_of, attributes=attrs.get(e.name, ()))
            for e in entities
        ]
        relations, _ = self._relations(source, entities)
        verbs = self._verbs(relations)

        names = tuple(e.name for e in entities)
        return OntologyDraft(
            entities=tuple(entities),
            relations=relations,
            verbs=verbs,
            data_needs=names,
            method_needs=(),
            facets={"factual": names, "cognitive": (), "communication": ()},
        )
