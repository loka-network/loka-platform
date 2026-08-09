"""Workflow A tests — build a KB spec from domain text, validated by the ontology loader."""

from __future__ import annotations

from loka_ontology import KeywordBuilder, OntologyEngine, build, load_ontology_str

_TEXT = (
    "The Central Bank sets the Policy Rate. A change in the Policy Rate affects the "
    "Exchange Rate, which in turn affects GDP. Analysts want to forecast GDP and compare "
    "policy options by estimating the effect of a rate change."
)


def test_build_produces_loadable_ontology() -> None:
    spec = build([_TEXT], KeywordBuilder())
    # The proposed ontology is valid (the loader disposed of nothing — it loads).
    onto = load_ontology_str(spec.ontology_yaml)
    engine = OntologyEngine(onto)
    for et in ("CentralBank", "PolicyRate", "ExchangeRate", "GDP"):
        assert engine.has_entity(et), f"expected entity {et}"


def test_build_splits_data_and_method_needs() -> None:
    spec = build([_TEXT], KeywordBuilder())
    # DATA needs = the entity types the ontology requires.
    assert "GDP" in spec.data_needs
    # METHODS needs = detected computations (forecast / compare->rank / effect->causal_effect).
    assert "forecast" in spec.method_needs
    assert "causal_effect" in spec.method_needs
    # Facets are populated (Factual / Cognitive).
    assert spec.facets["factual"]
    assert spec.facets["cognitive"]


def test_default_builder_is_keyword() -> None:
    # build() with no builder uses the deterministic reference (no LLM, reproducible).
    spec = build([_TEXT])
    assert spec.ontology_yaml.startswith("version:")
