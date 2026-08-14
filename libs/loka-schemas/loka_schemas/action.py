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
    status: str  # proposed | required | blocked
    requires_confirmation: bool = True
    blocked_by: str | None = None  # hard-constraint name if the G3 gate blocked it

    # --- deontic status: N(s, ac) ---
    # Separate from ``status``, which says what the chain is doing with this action. A norm says
    # what the action's standing is regardless. Two values (act / do not act) cannot express an
    # obligation, so this is three-valued.
    deontic_status: str = "permitted"  # permitted | mandatory | forbidden
    norm: str | None = None  # the norm that decided it, when one spoke
    # True when a norm makes this action mandatory in the current state. Then NOT acting is the
    # violation — the case a permitted/forbidden model has no way to report.
    omission_violates: bool = False
    # A state in which one norm obliges this action and another forbids it. Not resolved here:
    # picking a side would hide that the ontology contradicts itself, which is a defect in the
    # norms and belongs in front of whoever wrote them.
    normative_conflict: tuple[str, ...] = ()
