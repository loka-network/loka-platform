"""loka_grounding — natural language question -> typed query q* (G1).

Two stages: a proposer (Stage ①) turns a question into a candidate ``QueryProposal``; the
binder (Stage ②) validates it against the ontology into a signed ``TypedQuery``. The LLM only
proposes; the type system disposes.
"""

from .binder import bind, ground
from .models import (
    TASK_TYPES,
    EmptyProposal,
    GroundingError,
    QueryProposal,
    UnknownAttribute,
    UnknownTarget,
    UnknownTaskType,
)
from .proposer import KeywordProposer, QueryProposer

__all__ = [
    "ground",
    "bind",
    "QueryProposal",
    "QueryProposer",
    "KeywordProposer",
    "TASK_TYPES",
    "GroundingError",
    "EmptyProposal",
    "UnknownTaskType",
    "UnknownTarget",
    "UnknownAttribute",
]
