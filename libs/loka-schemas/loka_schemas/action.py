"""ActionProposal — a governed, not-yet-executed action (the Action layer's output).

An ontology ``ActionType`` (guard + effect) becomes a runtime ``ActionProposal`` when the chain
considers acting: the guard is evaluated against the world state, the hard-constraint gate (G3)
is applied, and — like Palantir's Action layer — execution requires confirmation. Nothing is
executed here; the proposal is what a human or a downstream executor approves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionProposal:
    """A proposed governed action awaiting confirmation/execution."""

    action_name: str
    verb: str
    target: str
    guard: str
    guard_status: str  # satisfied | not_satisfied | unverified
    effect: str
    status: str  # proposed | blocked
    requires_confirmation: bool = True
    blocked_by: str | None = None  # hard-constraint name if the G3 gate blocked it
