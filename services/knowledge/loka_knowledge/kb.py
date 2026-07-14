"""Evidence & provenance layer Kt — stores evidence, synthesises it, records contradictions.

Synthesis supports two pooling methods and a hierarchical (per-context) view:
  - fixed-effects: inverse-variance weighting (weights = 1/se²).
  - random-effects (DerSimonian-Laird): adds between-study heterogeneity τ², widening the
    pooled interval when studies disagree. Heterogeneity is reported as I².
  - hierarchical: when evidence spans multiple contexts (regimes/jurisdictions), each context
    is pooled separately and the per-context posteriors are retained, not collapsed.

When two records disagree beyond sampling error a contradiction record is opened — the
disagreement is surfaced, not silently averaged away.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from loka_schemas import (
    ContradictionRecord,
    DisagreementType,
    EffectDistribution,
    EvidenceRecord,
)

_Z = 1.96  # 95% threshold for "disagree beyond sampling error"
_EPS = 1e-9


class KnowledgeError(Exception):
    """Invalid knowledge-base operation (e.g. synthesising a claim with no evidence)."""


@dataclass(frozen=True)
class SynthesisResult:
    """Pooled estimate for a claim, plus heterogeneity, per-context views, and contradictions.

    Disagreement is preserved: contradictions are surfaced and per-context posteriors are kept.
    """

    claim_id: str
    pooled: EffectDistribution  # random-effects pooled estimate (heterogeneity-aware)
    fixed: EffectDistribution  # fixed-effects pooled estimate (for comparison)
    i_squared: float  # heterogeneity, 0..1 (0 = homogeneous)
    n_records: int
    per_context: dict[str, EffectDistribution] = field(default_factory=dict)
    contradictions: tuple[ContradictionRecord, ...] = ()

    @property
    def has_unresolved_contradiction(self) -> bool:
        return any(not c.resolved for c in self.contradictions)

    @property
    def is_heterogeneous(self) -> bool:
        return self.i_squared > 0.5


class KnowledgeBase:
    """Kt. Holds evidence records per claim and the contradictions opened between them."""

    def __init__(self) -> None:
        self._evidence: dict[str, list[EvidenceRecord]] = defaultdict(list)
        self._contradictions: list[ContradictionRecord] = []

    def add_evidence(self, record: EvidenceRecord) -> None:
        self._evidence[record.claim_id].append(record)

    def evidence_for(self, claim_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence.get(claim_id, []))

    def contradictions(self) -> tuple[ContradictionRecord, ...]:
        return tuple(self._contradictions)

    def synthesize(self, claim_id: str) -> SynthesisResult:
        """Pool a claim's evidence (fixed + random effects, per-context) and flag conflicts."""
        records = self._evidence.get(claim_id, [])
        if not records:
            raise KnowledgeError(f"no evidence for claim {claim_id}")

        estimates = [r.estimate for r in records]
        fixed, _ = _fixed_effects(estimates)
        random, i_squared = _random_effects(estimates)

        per_context: dict[str, EffectDistribution] = {}
        by_context: dict[str, list[EffectDistribution]] = defaultdict(list)
        for r in records:
            by_context[r.context or "(unspecified)"].append(r.estimate)
        for ctx, ests in by_context.items():
            per_context[ctx], _ = _random_effects(ests)

        found = self._detect_contradictions(claim_id, records)
        self._contradictions.extend(found)
        return SynthesisResult(
            claim_id=claim_id,
            pooled=random,
            fixed=fixed,
            i_squared=i_squared,
            n_records=len(records),
            per_context=per_context,
            contradictions=tuple(found),
        )

    @staticmethod
    def _detect_contradictions(
        claim_id: str, records: list[EvidenceRecord]
    ) -> list[ContradictionRecord]:
        out: list[ContradictionRecord] = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                diff = abs(a.estimate.mean - b.estimate.mean)
                combined_se = math.sqrt(a.estimate.se**2 + b.estimate.se**2)
                if diff > _Z * combined_se:
                    sign_flip = (a.estimate.mean > 0) != (b.estimate.mean > 0)
                    dtype = (
                        DisagreementType.EFFECT_SIGN
                        if sign_flip
                        else DisagreementType.EFFECT_MAGNITUDE
                    )
                    out.append(
                        ContradictionRecord(
                            contradiction_id=f"{claim_id}:{a.evidence_id}:{b.evidence_id}",
                            claim_id=claim_id,
                            evidence_a=a.evidence_id,
                            evidence_b=b.evidence_id,
                            disagreement_type=dtype,
                            detail=(
                                f"|{a.estimate.mean:.3f} - {b.estimate.mean:.3f}| "
                                f"= {diff:.3f} > {_Z} * {combined_se:.3f}"
                            ),
                        )
                    )
        return out


def _fixed_effects(estimates: list[EffectDistribution]) -> tuple[EffectDistribution, float]:
    """Inverse-variance fixed-effects pool. Returns (pooled, sum_of_weights)."""
    weights = [1.0 / max(e.se, _EPS) ** 2 for e in estimates]
    total_w = sum(weights)
    mean = sum(w * e.mean for w, e in zip(weights, estimates, strict=True)) / total_w
    return EffectDistribution(mean=mean, se=math.sqrt(1.0 / total_w)), total_w


def _random_effects(estimates: list[EffectDistribution]) -> tuple[EffectDistribution, float]:
    """DerSimonian-Laird random-effects pool. Returns (pooled, I²)."""
    k = len(estimates)
    if k == 1:
        return estimates[0], 0.0

    fixed, total_w = _fixed_effects(estimates)
    weights = [1.0 / max(e.se, _EPS) ** 2 for e in estimates]
    q = sum(w * (e.mean - fixed.mean) ** 2 for w, e in zip(weights, estimates, strict=True))
    df = k - 1
    c = total_w - sum(w**2 for w in weights) / total_w
    tau2 = max(0.0, (q - df) / c) if c > _EPS else 0.0
    i_squared = max(0.0, (q - df) / q) if q > _EPS else 0.0

    re_weights = [1.0 / (max(e.se, _EPS) ** 2 + tau2) for e in estimates]
    total_rw = sum(re_weights)
    mean = sum(w * e.mean for w, e in zip(re_weights, estimates, strict=True)) / total_rw
    return EffectDistribution(mean=mean, se=math.sqrt(1.0 / total_rw)), i_squared
