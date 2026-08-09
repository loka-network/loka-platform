"""Simulation stage (S4 Agent Society + S5 EcoFormer) — STUB.

Real version: the Manager plans a task DAG, an agent society simulates it tick by tick under
typed ActionTokens, and EcoFormer produces calibrated quantile outcomes; the scenario tree is
pruned and its leaves classified. This stub returns a fixed nominal + adverse pair derived from
the compiled world model so the end-to-end skeleton walks. It is deliberately NOT real yet.
"""

from __future__ import annotations

from loka_schemas import Scenario, ScenarioWorldModel


def simulate(wqt: ScenarioWorldModel) -> list[Scenario]:
    """STUB: return a nominal + adverse scenario for the compiled world model W(q, t)."""
    targets = ", ".join(wqt.state_package.entities) or "the queried entities"
    return [
        Scenario(
            scenario_id=f"{wqt.query_id}::nominal",
            kind="nominal",
            actions=(),
            outcome={"summary": f"baseline path for {targets}"},
            prob=0.6,
        ),
        Scenario(
            scenario_id=f"{wqt.query_id}::adverse",
            kind="adverse",
            actions=(),
            outcome={"summary": f"adverse-shock path for {targets}"},
            prob=0.4,
        ),
    ]
