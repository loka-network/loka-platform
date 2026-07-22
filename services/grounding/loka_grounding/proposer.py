"""Stage ① — propose a candidate query structure from natural language.

``QueryProposer`` is the port. ``KeywordProposer`` is a deterministic, LLM-free reference
implementation: it matches ontology entity-type names (and caller-supplied synonyms) as
substrings of the question and picks a task type by keyword. It is intentionally simple —
enough to run and test the pipeline offline. Real natural-language understanding is the
LLM proposer's job (see ``llm_proposer``); this one is the in-memory reference, mirroring the
engine's other reference implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .models import QueryProposal

# Keyword → task type, checked in order (first match wins). Counterfactual (an intervention,
# do(·)) outranks ranking so "if we fund A instead of B, which is better?" grounds as a
# counterfactual rather than a plain ranking.
_TASK_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("if ", "were to", "instead", "rather than", "counterfactual", "had we"), "counterfactual"),
    (("forecast", "will ", "expected", "next year", "project "), "conditional_forecast"),
    (("rank", "which ", "best", "most", "compare"), "ranking"),
)


@runtime_checkable
class QueryProposer(Protocol):
    """Turns a natural-language question into an unvalidated :class:`QueryProposal`."""

    def propose(self, question: str) -> QueryProposal: ...


class KeywordProposer:
    """Deterministic, no-LLM reference proposer.

    Matches entity-type names (lowercased) and any ``synonyms`` (mapping a lowercase phrase to
    an entity type) as substrings of the question. Entity order is preserved for determinism.
    """

    def __init__(
        self, entity_types: Sequence[str], synonyms: Mapping[str, str] | None = None
    ) -> None:
        self._entity_types = tuple(entity_types)
        self._synonyms = dict(synonyms or {})

    def _match_task(self, q: str) -> str:
        for phrases, task in _TASK_KEYWORDS:
            if any(p in q for p in phrases):
                return task
        return "descriptive"

    def propose(self, question: str) -> QueryProposal:
        q = question.lower()
        found: list[str] = []
        for et in self._entity_types:  # entity names first, in ontology order
            if et.lower() in q and et not in found:
                found.append(et)
        for phrase, et in self._synonyms.items():  # then synonyms
            if phrase in q and et not in found:
                found.append(et)
        return QueryProposal(
            task_type=self._match_task(q),
            targets=tuple(found),
            rationale="keyword match (reference proposer, no LLM)",
        )
