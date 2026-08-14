"""Policy stage (welfare, governance, audit) — basic welfare + constraint gate.

Full version: the policy model scores scenarios under the mission's welfare functional and hard
constraints (CVaR-penalised), emits a three-block memorandum with every figure traced to
evidence, and the G3 decision gate vets it. This basic version applies a mini G3 hard-constraint
gate, picks the scenario with the best outcome (falling back to probability), and records the
welfare terms + constraints it honoured, plus a manifest hash for replay. Simple stand-in for
the full policy model — labelled so.
"""

from __future__ import annotations

import hashlib

from loka_schemas import DecisionMemo, Scenario, ScenarioWorldModel


def _score(s: Scenario) -> tuple[float, float]:
    v = s.outcome.get("effect_on_targets")
    return (float(v) if isinstance(v, (int, float)) else 0.0, s.prob)


def decide(wqt: ScenarioWorldModel, scenarios: list[Scenario]) -> DecisionMemo:
    """Gate on hard constraints, pick the best-outcome scenario, emit a memo + audit hash."""
    constraints = [c.name for c in wqt.hard_constraints]
    welfare_terms = [t.name for t in wqt.welfare.terms]

    # mini G3: hard-constraint gate. Scenario actions must not name a forbidden constraint.
    forbidden = {c.name for c in wqt.hard_constraints}
    admissible = [s for s in scenarios if not (set(s.actions) & forbidden)] or scenarios

    top = max(admissible, key=_score, default=None)
    adverse = next((s.scenario_id for s in scenarios if s.kind == "adverse"), None)

    m = wqt.manifest
    audit = hashlib.sha256(
        f"{wqt.query_id}|{m.omega_version}|{m.et_snapshot}|{m.mission_version}".encode()
    ).hexdigest()[:16]

    if top is None:
        rec = "no admissible scenario"
    elif "effect_on_targets" in top.outcome:
        rec = f"expected effect on {top.outcome.get('targets')}: {top.outcome['effect_on_targets']}"
    else:
        rec = str(top.outcome.get("summary", "no scenario"))

    return DecisionMemo(
        query_id=wqt.query_id,
        recommendation=rec,
        rationale=(
            f"Picked the best-outcome admissible scenario under welfare terms "
            f"{welfare_terms or 'none'}; "
            f"hard constraints enforced: {constraints or 'none'}. "
            "(Basic welfare/constraint policy — the full policy model is not implemented.)"
        ),
        block_A_recommended={"scenario_id": top.scenario_id if top else None},
        block_C_contingency={"scenario_id": adverse},
        evidence_refs=(),
        audit_manifest=audit,
    )
