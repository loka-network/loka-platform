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
    signature: str | None = None
