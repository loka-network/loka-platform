"""Query dispatch — the DATA/METHODS split of the professor's query workflow (slide 7).

A typed query q* is routed to one of two branches:

  - ``asks``  (DATA)   -> retrieve(d from KB.DATA):  return the state slice already pulled into
                          W(q,t).  This is the Text2SQL / retrieval branch.
  - ``orders`` (METHOD) -> retrieve(m from KB.METHODS); apply m:  run a registered method over
                          W(q,t) — including its causal slice Γ(q) — and return the result.

The method here (`causal_effect`) reads the real causal slice, so the answer carries honest
causal effects with their identification status, not a free-text guess. If nothing applies, the
dispatch returns ``"don't know"`` (informs(li, sp, "don't know")), never a fabricated answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loka_schemas import Method, ScenarioWorldModel, TypedQuery

# task_type -> (speech act, target kind, method name | None)
_ACT_BY_TASK: dict[str, tuple[str, str, str | None]] = {
    "descriptive": ("asks", "data", None),
    "conditional_forecast": ("orders", "method", "causal_effect"),
    "counterfactual": ("orders", "method", "causal_effect"),
    "ranking": ("orders", "method", "causal_effect"),
}


def _causal_effect(wqt: ScenarioWorldModel) -> dict[str, Any]:
    """METHOD: report the causal chain reaching the query targets, with identification status."""
    sl = wqt.causal_slice
    if sl is None or not sl.claims:
        return {"answer": "don't know", "reason": "no causal path in Γ(q) for these targets"}
    effects = [
        {
            "cause": c.cause,
            "effect": c.effect,
            "mean": c.effect_distribution.mean,
            "se": c.effect_distribution.se,
            "identification_status": str(c.identification_status),
            "layer": str(c.layer),
            "evidence_refs": list(c.evidence_refs),
        }
        for c in sl.claims
    ]
    return {"answer": "causal_effect", "targets": list(sl.targets), "effects": effects}


_REGISTRY: dict[str, tuple[Method, Callable[[ScenarioWorldModel], dict[str, Any]]]] = {
    "causal_effect": (
        Method(
            name="causal_effect",
            description="Report the causal chain reaching the targets with identification status.",
            in_types=("targets",),
            out_type="effect_report",
        ),
        _causal_effect,
    ),
}


def resolve(q_star: TypedQuery, wqt: ScenarioWorldModel) -> dict[str, Any]:
    """Route q* to KB.DATA (asks) or KB.METHODS (orders) and return a structured result."""
    act, kind, method_name = _ACT_BY_TASK.get(q_star.task_type, ("asks", "data", None))
    if kind == "data":
        # asks -> retrieve(d from KB.DATA): the state slice already bound into W(q,t)
        facts = dict(wqt.state_package.state_slice)
        return {"act": act, "kind": kind, "method": None, "facts": facts}
    # orders -> retrieve(m from KB.METHODS); apply
    entry = _REGISTRY.get(method_name or "")
    if entry is None:
        return {"act": act, "kind": kind, "method": method_name, "result": {"answer": "don't know"}}
    _, fn = entry
    return {"act": act, "kind": kind, "method": method_name, "result": fn(wqt)}


def method_catalog() -> list[dict[str, Any]]:
    """The KB.METHODS catalogue (for introspection / a /methods endpoint later)."""
    return [
        {"name": m.name, "description": m.description,
         "in_types": list(m.in_types), "out_type": m.out_type}
        for m, _ in _REGISTRY.values()
    ]
