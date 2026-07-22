"""Types for the grounding pipeline: the Stage-① proposal and structured rejections.

A ``QueryProposal`` is the *unvalidated* candidate a proposer (keyword or LLM) produces
from a natural-language question. The binder (Stage ②) validates it against the ontology and
either returns a signed :class:`~loka_schemas.TypedQuery` or raises one of the errors here.
The split is deliberate: the LLM proposes, the type system disposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Task vocabulary the compiler understands. A proposal outside this set is rejected.
TASK_TYPES: frozenset[str] = frozenset(
    {"descriptive", "conditional_forecast", "counterfactual", "ranking"}
)


@dataclass(frozen=True)
class QueryProposal:
    """Stage-① output: a candidate structure proposed from a natural-language question.

    Nothing here is trusted yet — ``targets`` are raw names that may or may not resolve to
    ontology entity types, and ``task_type`` may be out of vocabulary. The binder checks both.
    """

    task_type: str
    targets: tuple[str, ...]
    rationale: str = ""
    unresolved: tuple[str, ...] = field(default_factory=tuple)  # names the proposer couldn't map


class GroundingError(Exception):
    """Base class for grounding failures — structured, never a bare string match."""


class EmptyProposal(GroundingError):
    """The proposer produced no candidate targets to ground."""


class UnknownTaskType(GroundingError):
    """The proposed task type is not in :data:`TASK_TYPES`."""


class UnknownTarget(GroundingError):
    """A proposed target does not resolve to a known ontology entity type."""
