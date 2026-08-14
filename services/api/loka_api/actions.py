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


def propose_actions(world: Any, wqt: ScenarioWorldModel) -> list[ActionProposal]:
    """Evaluate the ontology's action types against W(q,t); return governed proposals."""
    getter = getattr(world.engine, "action_types", None)
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

        status = "blocked" if (blocked or guard_status == "not_satisfied") else "proposed"
        proposals.append(
            ActionProposal(
                action_name=a.name,
                verb=a.verb,
                target=a.target,
                guard=a.guard,
                guard_status=guard_status,
                effect=a.effect,
                status=status,
                requires_confirmation=True,
                blocked_by=blocked,
            )
        )
    return proposals
