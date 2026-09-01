"""Stage ① — propose a candidate query structure from natural language.

``QueryProposer`` is the port. ``KeywordProposer`` is a deterministic, LLM-free reference
implementation: it matches ontology entity-type names (and caller-supplied synonyms) as
substrings of the question and picks a task type by keyword. It is intentionally simple —
enough to run and test the pipeline offline. Real natural-language understanding is the
LLM proposer's job (see ``llm_proposer``); this one is the in-memory reference, mirroring the
engine's other reference implementations.
"""

from __future__ import annotations

import re

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


#: A token written the way a field is: lower_snake_case. In a question, such a token is
#: almost always the name of an attribute being asked for rather than ordinary prose.
_FIELD_SHAPED = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


class KeywordProposer:
    """Deterministic, no-LLM reference proposer.

    Matches entity-type names (lowercased) and any ``synonyms`` (mapping a lowercase phrase to
    an entity type) as substrings of the question. Entity order is preserved for determinism.
    """

    def __init__(
        self,
        entity_types: Sequence[str],
        synonyms: Mapping[str, str] | None = None,
        attributes: Sequence[str] = (),
    ) -> None:
        self._entity_types = tuple(entity_types)
        self._synonyms = dict(synonyms or {})
        # Every attribute the ontology declares, so a question naming one can say which. What
        # this proposer cannot do is notice an attribute the ontology does *not* declare — it
        # matches against a list, so an unknown name simply fails to match. Refusing that case
        # is the LLM proposer's, and the binder checks whichever proposed it.
        self._attributes = tuple(attributes)

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
        asked = [
            a for a in self._attributes if a.lower() in q or a.replace("_", " ").lower() in q
        ]
        # Also anything written the way a field is written. Without this the proposer can only
        # ever name attributes the ontology declares, so a question asking for one it does not
        # would resolve to the entity and quietly return everything else about it — the failure
        # the binder's check exists to prevent, made unreachable by the proposer.
        for token in _FIELD_SHAPED.findall(q):
            if token not in asked:
                asked.append(token)
        return QueryProposal(
            task_type=self._match_task(q),
            targets=tuple(found),
            attributes=tuple(asked),
            rationale="keyword match (reference proposer, no LLM)",
        )
