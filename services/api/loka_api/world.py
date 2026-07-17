"""The configured world a deployment serves.

A ``World`` bundles the engine (Ω), state (Eₜ), signed mission, and causal graph that queries
are compiled against, held behind the port interfaces so backends are swappable.

- ``build_default_world`` — self-contained in-memory world (zero config).
- ``build_world_from_env`` — if ``NEO4J_URI`` and ``LOKA_PG_DSN`` are set, use the real
  backends (Neo4j causal graph + Postgres-fed state); otherwise fall back to in-memory.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from loka_ontology import OntologyEngine, load_ontology_str
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    CausalSlicer,
    EffectDistribution,
    HardConstraint,
    IdentificationStatus,
    MemoryAdapter,
    MissionProfile,
    OntologyView,
    StateView,
    TypedPredicate,
    WelfareFunctional,
    WelfareTerm,
)
from loka_state import WorldState


@dataclass
class World:
    """What a deployment is configured with. Fields are ports, so backends are swappable."""

    engine: OntologyView
    state: StateView
    mission: MissionProfile
    causal: CausalSlicer | None = None
    backend: str = "in-memory"


_ONTOLOGY = """
version: demo-v1
entities:
  - type: MacroIndicator
    properties:
      - {name: unit, type: string}
  - type: GDP
    subtype_of: MacroIndicator
    properties:
      - {name: value, type: double, required: true}
  - {type: CentralBank}
  - {type: PolicyLever}
verbs:
  - {name: RATE_CHANGE, class: institutional}
relations:
  - {name: sets, from: CentralBank, to: PolicyLever, cardinality: one_to_many}
"""


def _demo_mission() -> MissionProfile:
    return MissionProfile(
        version="demo-mission-v1",
        mandate="imported-inflation moderation with output-gap secondary",
        welfare=WelfareFunctional(
            terms=(WelfareTerm("inflation_dev", 0.7), WelfareTerm("output_gap", 0.3))
        ),
        hard_constraints=(HardConstraint("no_capital_controls", "forbidden in jurisdiction"),),
        signature="signed-by-ministry",
    )


def _demo_claim(
    cid: str,
    cause: str,
    effect: str,
    status: IdentificationStatus,
    layer: CausalLayer = CausalLayer.STRUCTURAL,
) -> CausalClaim:
    return CausalClaim(
        claim_id=cid,
        cause=cause,
        effect=effect,
        effect_distribution=EffectDistribution(mean=-1.0, se=0.3),
        identification_status=status,
        layer=layer,
    )


def build_default_world() -> World:
    """In-memory demo world (zero-config)."""
    from loka_causal import CausalGraph

    now = datetime(2026, 3, 18, tzinfo=UTC)
    engine = OntologyEngine(load_ontology_str(_ONTOLOGY))

    state = WorldState()
    state.set("GDP.TH.value", 2.1, now)
    state.set("GDP.TH.unit", "pct_yoy", now)
    state.set("CentralBank.Fed.policy_rate", 0.0525, now)

    graph = CausalGraph()
    graph.add_claim(_demo_claim("c1", "PolicyRate", "DXY", IdentificationStatus.STRUCTURAL))
    graph.add_claim(_demo_claim("c2", "DXY", "FX_EM", IdentificationStatus.STRUCTURAL))
    graph.add_claim(
        _demo_claim(
            "c3", "FX_EM", "GDP", IdentificationStatus.QUASI_EXPERIMENTAL, CausalLayer.EMPIRICAL
        )
    )
    return World(
        engine=engine, state=state, mission=_demo_mission(), causal=graph, backend="in-memory"
    )


def build_world_from_env() -> World:
    """Use real backends if configured (NEO4J_URI + LOKA_PG_DSN); else in-memory."""
    neo4j_uri = os.environ.get("NEO4J_URI")
    pg_dsn = os.environ.get("LOKA_PG_DSN")
    if not (neo4j_uri and pg_dsn):
        return build_default_world()

    from loka_adapters.postgres import PostgresAdapter
    from loka_causal.neo4j_graph import Neo4jCausalGraph

    engine = OntologyEngine(load_ontology_str(_ONTOLOGY))
    causal = Neo4jCausalGraph.connect(
        neo4j_uri,
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "loka_password"),
    )
    state = WorldState()
    adapter = PostgresAdapter(pg_dsn, adapter_id="pg", tables={"GDP": "gdp_state"})
    try:
        asyncio.run(_ingest_gdp(state, adapter))
    except Exception as exc:  # noqa: BLE001 - stay up even if not seeded yet
        print(f"[world] state ingest skipped: {exc}")
    return World(
        engine=engine,
        state=state,
        mission=_demo_mission(),
        causal=causal,
        backend="neo4j+postgres",
    )


async def _ingest_gdp(state: WorldState, adapter: MemoryAdapter) -> None:
    from loka_schemas import Certificate

    session = await adapter.authenticate(Certificate(subject="api", scopes=frozenset({"GDP"})))
    await state.ingest_from(adapter, TypedPredicate("GDP"), session)
