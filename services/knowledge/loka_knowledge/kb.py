"""Evidence & provenance layer Kt — stores evidence, synthesises it, records contradictions.

Synthesis is inverse-variance (fixed-effects) meta-analysis: the pooled estimate weights each
record by 1/se². When two records disagree beyond sampling error a contradiction record is
opened — the disagreement is surfaced, not silently averaged away.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

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
    """Pooled estimate for a claim, plus any contradictions found. Disagreement is preserved."""

    claim_id: str
    pooled: EffectDistribution
    n_records: int
    contradictions: tuple[ContradictionRecord, ...]

    @property
    def has_unresolved_contradiction(self) -> bool:
        return any(not c.resolved for c in self.contradictions)


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
        """Pool the evidence for a claim and open contradiction records for conflicts."""
        records = self._evidence.get(claim_id, [])
        if not records:
            raise KnowledgeError(f"no evidence for claim {claim_id}")

        pooled = _inverse_variance_pool([r.estimate for r in records])
        found = self._detect_contradictions(claim_id, records)
        self._contradictions.extend(found)
        return SynthesisResult(
            claim_id=claim_id,
            pooled=pooled,
            n_records=len(records),
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


def _inverse_variance_pool(estimates: list[EffectDistribution]) -> EffectDistribution:
    """Fixed-effects meta-analysis: weight each estimate by 1/se²."""
    weights = [1.0 / max(e.se, _EPS) ** 2 for e in estimates]
    total_w = sum(weights)
    pooled_mean = sum(w * e.mean for w, e in zip(weights, estimates, strict=True)) / total_w
    pooled_se = math.sqrt(1.0 / total_w)
    return EffectDistribution(mean=pooled_mean, se=pooled_se)
