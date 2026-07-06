"""World Model Compiler — binds Ω + Eₜ + Mission + q* into W(q, t).

This is the convergence point of the foundation: the ontology (S1), world state (S1), and
mission (S1) are bound into a single Scenario World Model. The causal slice Γ(q) is left
empty here and filled by the causal engine (S2).

Determinism: identical (ontology version, state snapshot, mission, query) inputs compile to
an equal W(q, t) — the basis for replay.
"""

from __future__ import annotations

from loka_ontology import OntologyEngine
from loka_schemas import (
    ManifestPins,
    MissionProfile,
    ScenarioStatePackage,
    ScenarioWorldModel,
    TypedQuery,
)
from loka_state import WorldState


class CompileError(Exception):
    """Base class for compilation failures."""


class MissionNotSigned(CompileError):
    """The mission profile must be signed before it can drive a compilation."""


class UnknownEntity(CompileError):
    """A query target does not resolve to a known ontology entity type."""


def compile_wqt(
    engine: OntologyEngine,
    state: WorldState,
    mission: MissionProfile,
    query: TypedQuery,
    *,
    scenario_id: str,
) -> ScenarioWorldModel:
    """Compile a Scenario World Model W(q, t). Causal slice left empty for S2."""
    if not mission.is_signed:
        raise MissionNotSigned("mission profile is not signed")

    unknown = [t for t in query.targets if not engine.has_entity(t)]
    if unknown:
        raise UnknownEntity(f"query targets not in ontology: {unknown}")

    state_package = ScenarioStatePackage(
        entities=query.targets,
        state_slice=state.slice(query.targets),
    )
    manifest = ManifestPins(
        omega_version=engine.version,
        et_snapshot=state.snapshot_hash(),
        mission_version=mission.version,
    )
    return ScenarioWorldModel(
        scenario_id=scenario_id,
        query_id=query.query_id,
        state_package=state_package,
        welfare=mission.welfare,
        hard_constraints=mission.hard_constraints,
        manifest=manifest,
        causal_slice=None,  # Γ(q) filled by the causal engine (S2)
    )
