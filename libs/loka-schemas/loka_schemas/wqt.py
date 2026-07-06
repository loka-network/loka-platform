"""W(q, t) — the Scenario World Model contract.

The single per-question object every downstream engine reads. Produced once by the compiler
from Ω + Eₜ + Mission (+ the causal slice Γ(q), which S2 fills in later). Version pins make a
run reproducible/replayable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .mission import HardConstraint, WelfareFunctional


@dataclass(frozen=True)
class ManifestPins:
    """Version pins bound into W(q, t) so a run can be replayed exactly (doc §3.1.6)."""

    omega_version: str  # ontology version
    et_snapshot: str  # hash of the world-state snapshot used
    mission_version: str  # mission profile version


@dataclass(frozen=True)
class ScenarioStatePackage:
    """The entity slice and relevant state values the question needs."""

    entities: tuple[str, ...]
    state_slice: Mapping[str, object]


@dataclass(frozen=True)
class ScenarioWorldModel:
    """W(q, t). The ``causal_slice`` is empty until S2 fills Γ(q)."""

    scenario_id: str
    query_id: str
    state_package: ScenarioStatePackage
    welfare: WelfareFunctional
    hard_constraints: tuple[HardConstraint, ...]
    manifest: ManifestPins
    causal_slice: object | None = None  # Γ(q); filled by the causal engine (S2)
