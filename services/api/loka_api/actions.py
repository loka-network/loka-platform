"""Action layer — turn the ontology's action types into governed proposals (Palantir layer 4).

For each ``ActionType`` in Ω, evaluate its guard against the world state, apply the hard-constraint
gate (G3), and emit an ``ActionProposal`` that requires confirmation before execution. Nothing is
executed here — this is the governed boundary where a human/executor approves. The guard evaluator
handles simple numeric preconditions (``name > n``) and honestly reports ``unverified`` otherwise.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from typing import Any

from loka_schemas import ActionProposal, ScenarioWorldModel

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}
_GUARD_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


def _eval_guard(guard: str, state_slice: dict[str, object]) -> str:
    """satisfied | not_satisfied | unverified — simple numeric guards only, honest otherwise."""
    m = _GUARD_RE.match(guard or "")
    if not m:
        return "unverified"
    name, op, num = m.group(1).lower(), m.group(2), float(m.group(3))
    for key, value in state_slice.items():
        k = key.lower()
        if k.endswith("." + name) or k == name or name in k:
            try:
                return "satisfied" if _OPS[op](float(value), num) else "not_satisfied"  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "unverified"
    return "unverified"


def _blocked_by(verb: str, target: str, hard_constraints: Any) -> str | None:
    """G3 gate: a hard constraint that forbids this action's verb/target blocks it."""
    for c in hard_constraints:
        text = f"{c.name} {c.description}".lower()
        if "forbid" in text and (verb.lower() in text or target.lower() in text):
            return str(c.name)
    return None


def _deontic_status(
    world: Any, action_name: str, state_slice: dict[str, object]
) -> tuple[str, str | None, tuple[str, ...]]:
    """N(s, ac) for one action: ``(status, deciding_norm, conflict)``.

    Only norms whose ``when`` holds in this state speak; an unconditional norm always speaks. A
    norm whose condition cannot be evaluated is treated as silent rather than as holding — an
    obligation asserted on an unreadable condition is not an obligation anyone can be held to,
    and inventing one here would put the system's own guesswork into a governance verdict.

    ``forbidden`` outranks ``mandatory`` when both hold, but the pair is reported rather than
    quietly resolved: a state that is at once obliged and forbidden is a defect in the norms.
    """
    norms_for = getattr(world.engine, "norms_for", None)
    if not callable(norms_for):
        return "permitted", None, ()

    speaking = [
        n for n in norms_for(action_name)
        if not n.when or _eval_guard(n.when, state_slice) == "satisfied"
    ]
    if not speaking:
        return "permitted", None, ()

    forbidding = [n for n in speaking if str(n.status) == "forbidden"]
    obliging = [n for n in speaking if str(n.status) == "mandatory"]
    conflict = (
        tuple(n.name for n in (*obliging, *forbidding)) if forbidding and obliging else ()
    )
    if forbidding:
        return "forbidden", forbidding[0].name, conflict
    if obliging:
        return "mandatory", obliging[0].name, conflict
    return "permitted", speaking[0].name, ()


def propose_actions(world: Any, wqt: ScenarioWorldModel) -> list[ActionProposal]:
    """Evaluate the ontology's action types against W(q,t); return governed proposals."""
    # Only Ac. An uncontrollable action is a change the environment makes; proposing one would
    # be proposing that the weather cooperate.
    getter = getattr(world.engine, "controllable_actions", None) or getattr(
        world.engine, "action_types", None
    )
    action_types = getter() if callable(getter) else []
    state = dict(wqt.state_package.state_slice)

    check = getattr(world.engine, "check_binding", None)

    proposals: list[ActionProposal] = []
    for a in action_types:
        guard_status = _eval_guard(a.guard, state)
        blocked = _blocked_by(a.verb, a.target, wqt.hard_constraints)

        # Ω's typing constraints (C) say which entity types a verb may act on at all. An action
        # whose target the constraints do not permit is not a governance decision to weigh — it
        # is not expressible, and proposing it and then refusing it later would be theatre.
        if blocked is None and callable(check):
            for actor in getattr(world.engine, "actor_types", lambda: [])() or [a.target]:
                verdict = check(a.verb, actor, a.target)
                if verdict.ok:
                    break
            else:
                blocked = f"type_constraint: {verdict.reason}"

        deontic, norm, conflict = _deontic_status(world, a.name, state)
        if deontic == "forbidden" and blocked is None:
            blocked = f"norm: {norm}"

        # An obligation whose guard does not hold is a state the system cannot act its way out
        # of: it must act and it cannot. That is worth surfacing, not smoothing over, so the
        # omission flag is set on the obligation itself rather than only where acting is possible.
        omission_violates = deontic == "mandatory" and blocked is None

        if blocked or guard_status == "not_satisfied":
            status = "blocked"
        elif deontic == "mandatory":
            status = "required"
        else:
            status = "proposed"

        proposals.append(
            ActionProposal(
                action_name=a.name,
                verb=a.verb,
                target=a.target,
                guard=a.guard,
                guard_status=guard_status,
                effect=a.effect,
                status=status,
                # A mandatory action still stops at the human boundary. An obligation says the
                # action must happen, not that this system may execute it unattended; the two
                # are separate decisions and collapsing them would let a norm grant autonomy.
                requires_confirmation=True,
                blocked_by=blocked,
                deontic_status=deontic,
                norm=norm,
                omission_violates=omission_violates,
                normative_conflict=conflict,
            )
        )
    return proposals
