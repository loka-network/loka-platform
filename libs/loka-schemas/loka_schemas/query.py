"""Typed query q* — the structured form of a customer question.

Minimal here; the grounding front-end will populate the full object (goal, intervention,
horizon, sufficiency, ...). What the compiler needs now: which entity types the question is
about, and an id/signature for replay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TypedQuery:
    """The typed, signed query object handed to the compiler. No free text."""

    query_id: str
    task_type: str  # e.g. "conditional_forecast", "counterfactual", ...
    targets: tuple[str, ...]  # ontology entity types the question is about
    #: The attributes the question asked for, validated against the targets that declare
    #: them. Empty means the question named none, which is not the same as naming one that
    #: does not exist — that is refused at binding and never reaches here.
    attributes: tuple[str, ...] = ()
    signature: str | None = None
