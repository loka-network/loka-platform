"""Tests for the evidence layer Kt: synthesis, variance reduction, contradiction detection."""

from __future__ import annotations

import pytest
from loka_knowledge import KnowledgeBase, KnowledgeError
from loka_schemas import (
    DisagreementType,
    EffectDistribution,
    EvidenceRecord,
    IdentificationStatus,
    StudyDesign,
)


def ev(eid: str, claim: str, mean: float, se: float) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        claim_id=claim,
        source=f"paper-{eid}",
        study_design=StudyDesign.DID,
        estimate=EffectDistribution(mean=mean, se=se),
        identification_status=IdentificationStatus.QUASI_EXPERIMENTAL,
    )


def test_synthesize_pools_concordant_evidence() -> None:
    kb = KnowledgeBase()
    for i, m in enumerate([-1.0, -1.1, -0.9]):
        kb.add_evidence(ev(f"e{i}", "c1", m, 0.3))
    res = kb.synthesize("c1")
    assert res.n_records == 3
    assert res.pooled.mean == pytest.approx(-1.0, abs=0.05)
    # inverse-variance pooling reduces the standard error below any single estimate
    assert res.pooled.se < 0.3
    assert not res.has_unresolved_contradiction


def test_synthesize_detects_sign_contradiction() -> None:
    kb = KnowledgeBase()
    kb.add_evidence(ev("a", "c2", 0.5, 0.1))
    kb.add_evidence(ev("b", "c2", -0.5, 0.1))
    res = kb.synthesize("c2")
    assert res.has_unresolved_contradiction
    assert res.contradictions[0].disagreement_type == DisagreementType.EFFECT_SIGN
    # the contradiction is also recorded in the knowledge base
    assert len(kb.contradictions()) == 1


def test_close_estimates_are_not_a_contradiction() -> None:
    kb = KnowledgeBase()
    kb.add_evidence(ev("a", "c3", -1.0, 0.3))
    kb.add_evidence(ev("b", "c3", -1.1, 0.3))
    res = kb.synthesize("c3")
    assert not res.has_unresolved_contradiction


def test_evidence_for_returns_all_records() -> None:
    kb = KnowledgeBase()
    kb.add_evidence(ev("a", "c4", -1.0, 0.2))
    kb.add_evidence(ev("b", "c4", -0.9, 0.2))
    assert len(kb.evidence_for("c4")) == 2


def test_synthesize_without_evidence_raises() -> None:
    kb = KnowledgeBase()
    with pytest.raises(KnowledgeError):
        kb.synthesize("missing")
