"""Causal knowledge contracts — the shape of causal claims and the slice that fills W(q, t).

A causal edge is not a scalar confidence; it is a typed claim with an effect distribution,
an identification status, and a layer. Downstream use is gated by an admissibility matrix
keyed on (identification status, use case) — the machinery lives in the causal service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IdentificationStatus(StrEnum):
    """How a causal effect was identified (doc Table 2). Governs admissible downstream use."""

    DEFINITIONAL = "definitional"  # true by accounting/market-clearing identity
    INSTITUTIONAL = "institutional"  # a rule enforced by authority, not mechanism
    STRUCTURAL = "structural"  # equilibrium/no-arbitrage/balance-sheet mechanism
    EXPERIMENTAL = "experimental"  # RCT / natural experiment
    QUASI_EXPERIMENTAL = "quasi_experimental"  # DiD / synthetic control / IV / RDD
    OBSERVATIONAL = "observational"  # correlational, no identification strategy
    EXPERT_PRIOR = "expert_prior"  # elicited from domain experts
    SIMULATOR_DERIVED = "simulator_derived"  # from Agent Society runs; quarantined
    PREDICTIVE_ONLY = "predictive_only"  # predictive but no causal claim


class CausalLayer(StrEnum):
    """Three-layer separation of Γ. Simulator-derived claims are confined to HYPOTHESIS."""

    STRUCTURAL = "structural_layer"  # identities, statutory rules, structural mechanisms
    EMPIRICAL = "empirical_layer"  # experimentally / quasi-experimentally identified
    HYPOTHESIS = "hypothesis_layer"  # observational / expert / simulator; awaiting review


class UseCase(StrEnum):
    """Downstream uses the admissibility matrix is keyed on (doc Table 3)."""

    HARD_CONSTRAINT = "hard_constraint"
    FORECAST_CONDITIONING = "forecast_conditioning"
    BLOCK_A_JUSTIFICATION = "block_a_justification"
    AUTHORITY_RESTRICTION = "authority_restriction"


@dataclass(frozen=True)
class EffectDistribution:
    """Posterior over effect size (normal, simplified): mean ± standard error."""

    mean: float
    se: float

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.mean - 1.96 * self.se, self.mean + 1.96 * self.se)


@dataclass(frozen=True)
class CausalClaim:
    """One typed causal edge cause → effect."""

    claim_id: str
    cause: str
    effect: str
    effect_distribution: EffectDistribution
    identification_status: IdentificationStatus
    layer: CausalLayer
    assumptions: tuple[str, ...] = ()
    context: str | None = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CausalSlice:
    """Γ(q) — the relevant causal subgraph bound into W(q, t) for a question's targets."""

    targets: tuple[str, ...]
    claims: tuple[CausalClaim, ...]
