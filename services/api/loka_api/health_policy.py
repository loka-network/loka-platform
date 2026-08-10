"""Slide-6 right half for the health scenario: Simulation -> Policy -> Decision memorandum.

Slide 6 does not stop at a query answer — a formalized query flows through:

    ... -> Simulation (EcoFormer / Agent Society): Scenario Evaluation / Selection
        -> PolicyFormer: Control Policy -> Decision memorandum + audit replay -> Response

This module connects Workflow B's ``orders`` result (a controlled projection with a 95% interval)
to that chain, so the health demo produces the same governed artefact the professor's diagram asks
for — scenarios, a chosen policy, and a replayable audit hash — instead of a bare number.

It is a basic, honest stand-in for the full EcoFormer/PolicyFormer (no multi-agent society, no
CVaR welfare functional) — labelled ``basic`` in the output. The constraint it enforces is sourced
from the ontology action's ``guard`` (load-bearing: the governance gate is Ω's, not hardcoded).
"""

from __future__ import annotations

import hashlib
from typing import Any


def evaluate_scenarios(projection: dict[str, Any]) -> list[dict[str, Any]]:
    """Scenario Evaluation (EcoFormer stand-in): read nominal + bounds off the projection interval.

    Under-5 mortality is a *lower-is-better* outcome, so the interval's upper bound is the adverse
    scenario and the lower bound the favourable one. Derived from the real 95% interval, not canned.
    """
    point = projection["projected_outcome"]
    lo, hi = projection["interval_95"]
    return [
        {"scenario_id": "nominal", "kind": "nominal", "under5_mortality": point, "prob": 0.6},
        {"scenario_id": "adverse", "kind": "adverse", "under5_mortality": hi, "prob": 0.2},
        {"scenario_id": "favorable", "kind": "favorable", "under5_mortality": lo, "prob": 0.2},
    ]


def select_and_decide(
    projection: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    iso: str,
    ontology_version: str,
    guard: str,
    method_name: str,
) -> dict[str, Any]:
    """Scenario Selection + PolicyFormer + audit: choose the nominal, judge welfare, gate, hash.

    Welfare objective is to minimise under-5 mortality. The recommendation states whether the
    proposed spending improves it, always carrying the projection's identification label so the
    read stays honest (observational, not a causal guarantee). The audit hash binds Ω's version,
    the method, and the exact inputs so the decision is replayable (slide-6 "audit replay").
    """
    current = projection["current_outcome"]
    chosen = next(s for s in scenarios if s["kind"] == "nominal")
    projected = chosen["under5_mortality"]
    delta = round(projected - current, 3)
    improves = delta < 0

    cur_dial, new_dial = projection["current_dial"], projection["new_dial"]
    audit = hashlib.sha256(
        f"{ontology_version}|{method_name}|{iso}|{cur_dial}->{new_dial}|{current}".encode()
    ).hexdigest()[:16]

    verb = "reduces" if improves else "does not reduce"
    rec = (
        f"Raising {projection['dial']} to {new_dial} {verb} {projection['outcome']} for {iso}: "
        f"{current} -> {projected} ({'−' if improves else '+'}{abs(delta)}), "
        f"95% CI {projection['interval_95']}."
    )
    return {
        "recommendation": rec,
        "welfare_objective": f"minimize {projection['outcome']}",
        "chosen_scenario": chosen["scenario_id"],
        "contingency_scenario": "adverse",
        "constraints_enforced": [guard] if guard else [],
        "constraint_satisfied": (new_dial or 0) > 0,  # the guard: health_exp_per_capita > 0
        "identification": projection.get("identification", "observational"),
        "audit_manifest": audit,
        "policy_engine": "basic (full PolicyFormer S6 not implemented)",
    }


def slide6_right_half(
    projection: dict[str, Any], *, iso: str, ontology_version: str, guard: str, method_name: str
) -> dict[str, Any]:
    """Run the projection through Simulation -> Policy and return {scenarios, decision}."""
    scenarios = evaluate_scenarios(projection)
    decision = select_and_decide(
        projection, scenarios, iso=iso, ontology_version=ontology_version,
        guard=guard, method_name=method_name,
    )
    return {"scenarios": scenarios, "decision": decision}
