"""The configured world a deployment serves.

A ``World`` bundles the engine (Ω), state (Eₜ), signed mission, and causal graph that queries
are compiled against. It holds them behind the port interfaces, so a production deployment
swaps in real backends (Neo4jCausalGraph, Postgres-backed state) without changing the API.

``build_default_world`` returns a self-contained in-memory world so the service runs with zero
configuration; production wires real backends here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from loka_causal import CausalGraph
from loka_ontology import OntologyEngine, load_ontology_str
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    CausalSlicer,
    EffectDistribution,
    HardConstraint,
    IdentificationStatus,
    MissionProfile,
    OntologyView,
    StateView,
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


def build_default_world() -> World:
    """In-memory demo world (zero-config). Production replaces the backends here."""
    now = datetime(2026, 3, 18, tzinfo=UTC)

    engine = OntologyEngine(load_ontology_str(_ONTOLOGY))

    state = WorldState()
    state.set("GDP.TH.value", 2.1, now)
    state.set("GDP.TH.unit", "pct_yoy", now)
    state.set("CentralBank.Fed.policy_rate", 0.0525, now)

    mission = MissionProfile(
        version="demo-mission-v1",
        mandate="imported-inflation moderation with output-gap secondary",
        welfare=WelfareFunctional(
            terms=(WelfareTerm("inflation_dev", 0.7), WelfareTerm("output_gap", 0.3))
        ),
        hard_constraints=(HardConstraint("no_capital_controls", "forbidden in jurisdiction"),),
        signature="signed-by-ministry",
    )

    graph = CausalGraph()
    graph.add_claim(_claim("c1", "PolicyRate", "DXY", IdentificationStatus.STRUCTURAL))
    graph.add_claim(_claim("c2", "DXY", "FX_EM", IdentificationStatus.STRUCTURAL))
    graph.add_claim(
        _claim("c3", "FX_EM", "GDP", IdentificationStatus.QUASI_EXPERIMENTAL, CausalLayer.EMPIRICAL)
    )

    return World(engine=engine, state=state, mission=mission, causal=graph)


def _claim(
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
