"""Scenario — one branch of the scenario tree.

Produced by the simulation stage (S4 Agent Society + S5 EcoFormer). Every scenario is one
possible future for the queried world model, classified as nominal / adverse / mandated and
carried, with its probability, into the policy stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    """A single simulated future branch handed to the policy stage."""

    scenario_id: str
    kind: str  # "nominal" | "adverse" | "mandated"
    actions: tuple[str, ...] = ()
    outcome: Mapping[str, object] = field(default_factory=dict)
    prob: float = 0.0
