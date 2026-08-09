"""Simulation stage (S4 Agent Society + S5 EcoFormer) — basic, causal-driven.

Full version: a Manager plans a DAG, an agent society simulates it tick by tick, and EcoFormer
produces calibrated quantile outcomes. This basic version derives a nominal + adverse pair from
the *real* causal slice Γ(q): the effect reaching the targets sets the nominal outcome, and a
~5% downside (mean - 1.645·se) sets the adverse one. With no causal path it falls back to
qualitative canned scenarios. It is a simple stand-in, not the full simulator — labelled so.
"""

from __future__ import annotations

from loka_schemas import Scenario, ScenarioWorldModel


def simulate(wqt: ScenarioWorldModel) -> list[Scenario]:
    """Derive nominal/adverse scenarios from Γ(q) when present, else qualitative fallbacks."""
    targets = set(wqt.state_package.entities)
    sl = wqt.causal_slice
    direct = [c for c in sl.claims if c.effect in targets] if sl is not None else []

    if direct:
        mean = sum(c.effect_distribution.mean for c in direct)
        se = sum(c.effect_distribution.se ** 2 for c in direct) ** 0.5
        tgt = ", ".join(sorted(targets))
        return [
            Scenario(
                scenario_id=f"{wqt.query_id}::nominal",
                kind="nominal",
                outcome={"effect_on_targets": round(mean, 3), "targets": tgt},
                prob=0.6,
            ),
            Scenario(
                scenario_id=f"{wqt.query_id}::adverse",
                kind="adverse",
                outcome={"effect_on_targets": round(mean - 1.645 * se, 3), "targets": tgt},
                prob=0.4,
            ),
        ]

    label = ", ".join(wqt.state_package.entities) or "the queried entities"
    return [
        Scenario(
            scenario_id=f"{wqt.query_id}::nominal",
            kind="nominal",
            outcome={"summary": f"baseline path for {label}"},
            prob=0.6,
        ),
        Scenario(
            scenario_id=f"{wqt.query_id}::adverse",
            kind="adverse",
            outcome={"summary": f"adverse-shock path for {label}"},
            prob=0.4,
        ),
    ]
