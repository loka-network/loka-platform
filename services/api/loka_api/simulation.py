"""Simulation stage — basic, causal-driven.

Full version: a manager plans a DAG, an agent society simulates it tick by tick, and a
calibrated forecasting model
produces calibrated quantile outcomes. This basic version derives a nominal + adverse pair from
the *real* causal slice Γ(q): the effect reaching the targets sets the nominal outcome, and a
~5% downside (mean - 1.645·se) sets the adverse one. With no causal path it falls back to
qualitative canned scenarios. It is a simple stand-in, not the full simulator — labelled so.
"""

from __future__ import annotations

from typing import Any

from loka_schemas import Scenario, ScenarioWorldModel


def actor_reactions(
    engine: Any, scenario_summary: str, *, max_actors: int = 3
) -> dict[str, Any]:
    """How the entity types Ω permits to act would respond to a scenario.

    The actors are read from Ω's typing constraints — an entity type is an actor here because a
    constraint names it as the agent of a verb, not because this module lists it. Add a
    constraint and a new actor appears; remove it and one disappears.

    The reply carries the kind of engine that produced it. A general assistant standing in for
    the behavior model is agreeable by construction, so it under-produces exactly the refusals
    and delays a simulation exists to surface; a reader must be able to tell which one answered.
    """
    from loka_serving import Persona, persona_engine_for

    actors: list[str] = getattr(engine, "actor_types", lambda: [])()[:max_actors]
    if not actors:
        return {"actors": [], "engine": "none", "note": "Ω names no agent in any constraint"}

    reactions: list[dict[str, str]] = []
    kind = "none"
    for name in actors:
        persona = Persona(name=name, domain=name.lower())
        try:
            behavior, kind = persona_engine_for(persona)
            action = behavior.act(
                social_context=scenario_summary, persona=persona, history=[]
            )
        except Exception as exc:  # noqa: BLE001 - a simulation failure is not an answer failure
            reactions.append({"actor": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reactions.append({"actor": name, "action": action})

    return {
        "actors": reactions,
        "engine": kind,
        "calibrated": kind == "behavior-model",
        "note": (
            "produced by a general assistant standing in for the behavior model; it is "
            "cooperative by construction and under-states adversarial responses"
            if kind == "general-llm"
            else "deterministic placeholder, not behaviour" if kind == "stub" else ""
        ),
    }


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
