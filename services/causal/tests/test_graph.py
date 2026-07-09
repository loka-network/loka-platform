"""Tests for the causal graph Γ: queries, layer separation, quarantine, and slicing."""

from __future__ import annotations

import pytest
from loka_causal import CORE_LAYERS, CausalError, CausalGraph, build_slice
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    IdentificationStatus,
)

E = EffectDistribution(mean=-1.0, se=0.3)
S = IdentificationStatus

# default layer per identification status (tests can override)
_LAYER = {
    S.STRUCTURAL: CausalLayer.STRUCTURAL,
    S.EXPERIMENTAL: CausalLayer.EMPIRICAL,
    S.QUASI_EXPERIMENTAL: CausalLayer.EMPIRICAL,
    S.OBSERVATIONAL: CausalLayer.HYPOTHESIS,
    S.SIMULATOR_DERIVED: CausalLayer.HYPOTHESIS,
}


def claim(
    cid: str,
    cause: str,
    effect: str,
    status: IdentificationStatus,
    layer: CausalLayer | None = None,
) -> CausalClaim:
    return CausalClaim(
        claim_id=cid,
        cause=cause,
        effect=effect,
        effect_distribution=E,
        identification_status=status,
        layer=layer or _LAYER[status],
    )


def make_graph() -> CausalGraph:
    g = CausalGraph()
    # transmission chain: Fed.rate → DXY → USD/THB → ExporterMargin → TH.GDP
    g.add_claim(claim("c1", "Fed.rate", "DXY", S.STRUCTURAL))
    g.add_claim(claim("c2", "DXY", "USD/THB", S.STRUCTURAL))
    g.add_claim(claim("c3", "USD/THB", "ExporterMargin", S.QUASI_EXPERIMENTAL))
    g.add_claim(claim("c4", "ExporterMargin", "TH.GDP", S.QUASI_EXPERIMENTAL))
    # confounder: OilPrice causes both DXY and TH.GDP
    g.add_claim(claim("c5", "OilPrice", "DXY", S.STRUCTURAL))
    g.add_claim(claim("c6", "OilPrice", "TH.GDP", S.OBSERVATIONAL))
    return g


def test_ancestors_core_excludes_hypothesis() -> None:
    g = make_graph()
    anc = g.ancestors("TH.GDP", layers=CORE_LAYERS)
    assert anc == {"ExporterMargin", "USD/THB", "DXY", "Fed.rate", "OilPrice"}


def test_ancestors_all_layers_includes_hypothesis_paths() -> None:
    g = make_graph()
    # with all layers, OilPrice still reaches TH.GDP directly via the hypothesis edge c6
    assert "OilPrice" in g.ancestors("TH.GDP", layers=None)


def test_descendants() -> None:
    g = make_graph()
    assert g.descendants("Fed.rate", layers=CORE_LAYERS) == {
        "DXY",
        "USD/THB",
        "ExporterMargin",
        "TH.GDP",
    }


def test_mediators_are_the_transmission_chain() -> None:
    g = make_graph()
    assert g.mediators("Fed.rate", "TH.GDP", layers=CORE_LAYERS) == {
        "DXY",
        "USD/THB",
        "ExporterMargin",
    }


def test_confounders_finds_common_cause() -> None:
    g = make_graph()
    # OilPrice is a common cause of DXY and TH.GDP (across all layers)
    assert "OilPrice" in g.confounders("DXY", "TH.GDP", layers=None)


def test_simulator_derived_must_be_quarantined() -> None:
    g = CausalGraph()
    bad = claim("x", "Sentiment", "TH.GDP", S.SIMULATOR_DERIVED, CausalLayer.STRUCTURAL)
    with pytest.raises(CausalError):
        g.add_claim(bad)
    # allowed only in the hypothesis layer
    ok = claim("x", "Sentiment", "TH.GDP", S.SIMULATOR_DERIVED, CausalLayer.HYPOTHESIS)
    g.add_claim(ok)
    assert len(g.claims()) == 1


def test_build_slice_is_causal_core_only() -> None:
    g = make_graph()
    sl = build_slice(g, ["TH.GDP"])
    ids = {c.claim_id for c in sl.claims}
    assert ids == {"c1", "c2", "c3", "c4", "c5"}  # c6 (hypothesis) excluded
    assert sl.targets == ("TH.GDP",)
