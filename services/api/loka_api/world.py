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
from typing import Any

from loka_ontology import OntologyEngine, load_ontology, load_ontology_str
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    CausalSlicer,
    EffectDistribution,
    HardConstraint,
    IdentificationStatus,
    KBSpec,
    MemoryAdapter,
    MissionProfile,
    OntologyView,
    StateStore,
    TypedPredicate,
    WelfareFunctional,
    WelfareTerm,
)
from loka_state import WorldState


@dataclass
class World:
    """What a deployment is configured with. Fields are ports, so backends are swappable."""

    engine: OntologyView
    state: StateStore
    mission: MissionProfile
    causal: CausalSlicer | None = None
    # Kt — the evidence behind the causal claims. Populated by ingestion; kept beside Γ rather
    # than inside it, because a pooled estimate and the disagreement it papers over are two
    # different things a reader may need.
    knowledge: object | None = None
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
  # policy_rate is declared because CutPolicyLever's guard reads it. It was not, and the guard
  # was checked against a type with no attributes at all, so the act could never be proposed —
  # silently, since an act that never appears looks the same as an act whose moment has not come.
  - type: PolicyLever
    properties:
      - {name: policy_rate, type: double, description: "the lever's current setting, in percent"}
verbs:
  - {name: RATE_CHANGE, class: institutional}
relations:
  - {name: sets, from: CentralBank, to: PolicyLever, cardinality: one_to_many}
actions:
  - name: CutPolicyLever
    verb: RATE_CHANGE
    target: PolicyLever
    guard: "policy_rate > 0"
    effect: "policy_rate decreases by 25bp"
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


def world_from_kbspec(spec: KBSpec) -> World:
    """Build a queryable World from a Workflow-A KBSpec — closing the build->answer loop.

    The built ontology comes from ``spec.ontology_yaml``; state and causal start empty (Workflow
    A produces the ontology + DATA/METHODS needs, not yet the data or causal edges), and a signed
    default mission lets the compiler run. Queries against this world therefore ground and compile
    against the just-built ontology, with empty state/causal until those are ingested.
    """
    ontology_yaml = spec.ontology_yaml
    engine = OntologyEngine(load_ontology_str(ontology_yaml))
    return World(
        engine=engine,
        state=WorldState(),
        mission=_demo_mission(),
        causal=None,
        backend="built-kb",
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
        asyncio.run(_ingest_gdp(state, adapter, engine))
    except Exception as exc:  # noqa: BLE001 - stay up even if not seeded yet
        print(f"[world] state ingest skipped: {exc}")
    return World(
        engine=engine,
        state=state,
        mission=_demo_mission(),
        causal=causal,
        backend="neo4j+postgres",
    )


async def _ingest_gdp(state: WorldState, adapter: MemoryAdapter, engine: OntologyEngine) -> None:
    """Read the attributes Ω declares for the entity — not whatever columns the table happens
    to have. The ontology decides what an entity *is*, so it decides what is read."""
    from loka_schemas import Certificate

    session = await adapter.authenticate(Certificate(subject="api", scopes=frozenset({"GDP"})))
    predicate = TypedPredicate("GDP", columns=tuple(sorted(engine.properties_of("GDP"))))
    await state.ingest_from(adapter, predicate, session)


def build_supply_world() -> World:
    """The supply-chain ontology as a queryable world, with its rows in state.

    The query path is not tied to a domain: ``_formalize`` reads the entity types off
    ``world.engine`` and the binder validates a proposal against that same ontology, so a
    question is answered against whatever ontology the world carries. What was missing was not
    generality but registration — the supply ontology existed only behind its own endpoints, so
    nothing could ask it a question in natural language and the query chapter had to draw its
    examples from a different domain.

    State variables are named ``<EntityType>.<id>.<field>``, which is the form ``WorldState``
    slices on. The identifier comes from the entity's first required attribute, that being the
    ontology's own statement of what identifies one of these.
    """
    engine = OntologyEngine(load_ontology(_supply_ontology_path()))
    state = WorldState()
    now = datetime.now(UTC)

    from .supply import load_supply_dataset

    for entity, rows in load_supply_dataset(engine).items():
        key = _identity_attribute(engine, entity)
        if key is None:
            continue
        for row in rows:
            rid = row.get(key)
            if rid is None:
                continue
            for field_name, value in row.items():
                state.set(f"{entity}.{rid}.{field_name}", value, now)

    return World(
        engine=engine,
        state=state,
        mission=_demo_mission(),
        causal=None,
        backend="supply",
    )


def _identity_attribute(engine: Any, entity: str) -> str | None:
    """The attribute the ontology marks as required — what identifies one of these."""
    for name, prop in sorted(engine.properties_of(entity).items()):
        if prop.required:
            return name
    return None


def _supply_ontology_path() -> str:
    """The committed ontology, found relative to this file or to the working directory.

    Both, because they are different deployments. Running from a checkout, the file sits a few
    directories above this one. Installed — which is what the image does — this module lives in
    site-packages and no number of parent directories reaches the repository; there the working
    directory is where the ontology is. The first version walked upward only, so the supply
    world silently failed to register in the image while working everywhere it was tested.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, *([".."] * n), "examples", "supply_ontology.yaml")
        for n in range(1, 6)
    ]
    candidates.append(os.path.join(os.getcwd(), "examples", "supply_ontology.yaml"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "examples/supply_ontology.yaml not found, relative to this module or the working "
        f"directory ({os.getcwd()})"
    )


def build_roles_world() -> World:
    """The three-role contract as a queryable world.

    State is empty by design. Every condition these norms read — a ceiling, a sample size, a
    statistic — belongs to the run the caller is conducting, and is passed in with the act being
    checked. A value this service held would be one the caller could not account for, and the
    whole arrangement turns on numbers being traceable to whoever measured them.
    """
    engine = OntologyEngine(load_ontology(_find("roles_ontology.yaml")))
    return World(
        engine=engine,
        state=WorldState(),
        mission=_demo_mission(),
        causal=None,
        backend="roles",
    )


def _find(name: str) -> str:
    """A committed ontology, found relative to this module or to the working directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, *([".."] * n), "examples", name) for n in range(1, 6)
    ]
    candidates.append(os.path.join(os.getcwd(), "examples", name))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"examples/{name} not found, relative to this module or the working "
        f"directory ({os.getcwd()})"
    )
