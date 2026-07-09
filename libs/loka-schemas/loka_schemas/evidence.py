"""Evidence & provenance contracts (Kt) — why the system believes a causal claim.

Kt is organised around typed records, not documents: each causal claim is supported (or
challenged) by evidence records; when two disagree beyond sampling error, a contradiction
record is opened. Disagreement is preserved, never silently fused.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .causal import EffectDistribution, IdentificationStatus


class StudyDesign(StrEnum):
    """How an evidence estimate was produced (doc appendix F.3)."""

    RCT = "rct"
    NATURAL_EXPERIMENT = "natural_experiment"
    DID = "did"
    SYNTHETIC_CONTROL = "synthetic_control"
    IV = "iv"
    RDD = "rdd"
    STRUCTURAL_MODEL = "structural_model"
    OBSERVATIONAL = "observational"
    SIMULATION = "simulation"


class DisagreementType(StrEnum):
    """Why two evidence records conflict."""

    EFFECT_SIGN = "effect_sign"
    EFFECT_MAGNITUDE = "effect_magnitude"


@dataclass(frozen=True)
class EvidenceRecord:
    """One supporting estimate for a causal claim."""

    evidence_id: str
    claim_id: str
    source: str
    study_design: StudyDesign
    estimate: EffectDistribution
    identification_status: IdentificationStatus
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContradictionRecord:
    """Opened when two evidence records for the same claim disagree beyond sampling error."""

    contradiction_id: str
    claim_id: str
    evidence_a: str
    evidence_b: str
    disagreement_type: DisagreementType
    detail: str
    resolved: bool = False
