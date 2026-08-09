"""Workflow A — ontology generation from domain texts (Sifakis slide 7, left half).

    domain texts + prompt --LLM--> ontology + acquired knowledge (DATA / METHODS) --> KB

Same discipline as grounding: the builder *proposes* a draft (an ``OntologyBuilder``); the
type system *disposes* — :func:`build` compiles the draft into an ontology definition and runs
it through ``load_ontology_str``, so a malformed or inconsistent proposal is rejected, not
trusted. Two proposers ship: a deterministic ``KeywordBuilder`` (no LLM, for tests / offline /
sovereign runs) and an opt-in ``LLMBuilder``.
"""

from __future__ import annotations

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
        "Analysts", "Given",
    }
)
_TITLE = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b")
_METHOD_WORDS = {
    "forecast": "forecast",
    "predict": "forecast",
    "compare": "rank",
    "rank": "rank",
    "effect": "causal_effect",
    "impact": "causal_effect",
    "cause": "causal_effect",
}


@dataclass(frozen=True)
class OntologyDraft:
    """An unvalidated proposal from a builder (before the type system disposes)."""

    entity_types: tuple[str, ...]
    relations: tuple[tuple[str, str, str], ...] = ()  # (from, name, to)
    data_needs: tuple[str, ...] = ()
    method_needs: tuple[str, ...] = ()
    facets: dict[str, tuple[str, ...]] = field(default_factory=dict)


@runtime_checkable
class OntologyBuilder(Protocol):
    """Turns domain texts into an unvalidated :class:`OntologyDraft`."""

    def propose(self, texts: Sequence[str]) -> OntologyDraft: ...


def _camel(phrase: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in phrase.split())


class KeywordBuilder:
    """Deterministic, no-LLM reference builder.

    Entity types = distinct Title-Case noun phrases (CamelCased). Method needs = detected verbs
    (forecast / compare / effect ...). Facets: Factual = entities, Cognitive = method needs.
    """

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        blob = "\n".join(texts)
        entities: list[str] = []
        for phrase in _TITLE.findall(blob):
            words = phrase.split()
            while words and words[0] in _STOP:  # drop leading "The"/"A"/... from the phrase
                words.pop(0)
            if not words:
                continue
            name = _camel(" ".join(words))
            if name and name not in entities:
                entities.append(name)
        methods: list[str] = []
        low = blob.lower()
        for word, method in _METHOD_WORDS.items():
            if word in low and method not in methods:
                methods.append(method)
        return OntologyDraft(
            entity_types=tuple(entities),
            data_needs=tuple(entities),  # each entity type needs data
            method_needs=tuple(methods),
            facets={
                "factual": tuple(entities),
                "cognitive": tuple(methods),
                "communication": (),
            },
        )


class LLMBuilder:
    """Opt-in model-backed builder. Injectable client (any ``messages.create(...)``)."""

    _SYSTEM = (
        "Extract an ontology from the domain text. Reply with ONLY a JSON object: "
        '{"entities": [<CamelCase type names>], "relations": [[from, name, to], ...], '
        '"data_needs": [...], "method_needs": [...]}. No prose, no code fences.'
    )

    def __init__(self, *, client: Any | None = None, model: str = "claude-opus-4-8") -> None:
        if client is None:
            import anthropic  # optional [llm] extra, imported lazily

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        import json

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._SYSTEM,
            messages=[{"role": "user", "content": "\n".join(texts)}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1]) if start != -1 else {}
        rels = tuple(
            (r[0], r[1], r[2]) for r in obj.get("relations", []) if isinstance(r, list) and len(r) == 3
        )
        entities = tuple(dict.fromkeys(obj.get("entities", [])))
        return OntologyDraft(
            entity_types=entities,
            relations=rels,
            data_needs=tuple(obj.get("data_needs", [])) or entities,
            method_needs=tuple(obj.get("method_needs", [])),
            facets={"factual": entities, "cognitive": tuple(obj.get("method_needs", []))},
        )


def build(texts: Sequence[str], builder: OntologyBuilder | None = None) -> KBSpec:
    """Workflow A end-to-end: propose a draft, then compile + validate it into a KBSpec.

    Raises ``OntologyLoadError`` if the proposed ontology is inconsistent (the type system
    disposing of a bad proposal) — the caller sees a structured failure, never a silent bad KB.
    """
    builder = builder or KeywordBuilder()
    draft = builder.propose(texts)
    yaml = _draft_to_yaml(draft)
    load_ontology_str(yaml)  # validate: raises on inconsistency
    return KBSpec(
        ontology_yaml=yaml,
        data_needs=draft.data_needs,
        method_needs=draft.method_needs,
        facets={k: tuple(v) for k, v in draft.facets.items()},
    )


def _draft_to_yaml(draft: OntologyDraft) -> str:
    lines = ["version: built-v0.1", "entities:"]
    known = set(draft.entity_types)
    for et in draft.entity_types:
        lines.append(f"  - {{type: {et}}}")
    rels = [r for r in draft.relations if r[0] in known and r[2] in known]
    if rels:
        lines.append("relations:")
        for src, name, tgt in rels:
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "relates"
            lines.append(f"  - {{name: {safe}, from: {src}, to: {tgt}}}")
    return "\n".join(lines) + "\n"
