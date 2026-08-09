"""Policy stage (S6 PolicyFormer + governance/audit) — STUB.

Real version: PolicyFormer scores scenarios under the mission's welfare functional and hard
constraints (CVaR-penalised), emits a three-block decision memorandum (recommended / mandated /
contingency) with every figure traced to evidence, and the G3 decision gate vets it. This stub
picks the highest-probability scenario and fills the memorandum shell, plus a manifest hash so
the run is replayable. It is deliberately NOT real yet.
"""

from __future__ import annotations

import hashlib

from loka_schemas import DecisionMemo, Scenario, ScenarioWorldModel


def decide(wqt: ScenarioWorldModel, scenarios: list[Scenario]) -> DecisionMemo:
    """STUB: choose the top-probability scenario; return a memorandum shell + audit hash."""
    top = max(scenarios, key=lambda s: s.prob, default=None)
    adverse = next((s.scenario_id for s in scenarios if s.kind == "adverse"), None)
    constraints = "; ".join(c.name for c in wqt.hard_constraints) or "none"

    m = wqt.manifest
    audit = hashlib.sha256(
        f"{wqt.query_id}|{m.omega_version}|{m.et_snapshot}|{m.mission_version}".encode()
    ).hexdigest()[:16]

    recommendation = str(top.outcome.get("summary", "no scenario")) if top else "no scenario"
    return DecisionMemo(
        query_id=wqt.query_id,
        recommendation=recommendation,
        rationale=(
            "STUB policy: selected the highest-probability scenario. The mission's welfare "
            f"terms and hard constraints ({constraints}) are carried through from W(q,t) but "
            "not yet optimised — PolicyFormer (S6) is not implemented."
        ),
        block_A_recommended={"scenario_id": top.scenario_id if top else None},
        block_C_contingency={"scenario_id": adverse},
        evidence_refs=(),
        audit_manifest=audit,
    )
