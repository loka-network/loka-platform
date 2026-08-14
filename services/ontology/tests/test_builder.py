"""Workflow A tests — build a KB spec from domain text, validated by the ontology loader."""

from __future__ import annotations

import json
from types import SimpleNamespace

from loka_ontology import KeywordBuilder, LLMBuilder, OntologyEngine, build, load_ontology_str

_TEXT = (
    "The Central Bank sets the Policy Rate. A change in the Policy Rate affects the "
    "Exchange Rate, which in turn affects GDP. Analysts want to forecast GDP and compare "
    "policy options by estimating the effect of a rate change."
)


def test_build_produces_loadable_ontology() -> None:
    spec = build([_TEXT], KeywordBuilder())
    onto = load_ontology_str(spec.ontology_yaml)  # loader disposed of nothing -> it loads
    engine = OntologyEngine(onto)
    for et in ("CentralBank", "PolicyRate", "ExchangeRate", "GDP"):
        assert engine.has_entity(et), f"expected entity {et}"


def test_keyword_builder_extracts_relations_and_verbs() -> None:
    spec = build([_TEXT], KeywordBuilder())
    onto = load_ontology_str(spec.ontology_yaml)
    rel_pairs = {(r.from_type, r.to_type) for r in onto.relations}
    assert ("CentralBank", "PolicyRate") in rel_pairs  # "Central Bank sets the Policy Rate"
    assert onto.verbs, "expected action verbs extracted from the relation verbs"


def test_build_splits_data_and_method_needs() -> None:
    spec = build([_TEXT], KeywordBuilder())
    assert "GDP" in spec.data_needs
    assert "forecast" in spec.method_needs
    assert "causal_effect" in spec.method_needs
    assert spec.facets["factual"] and spec.facets["cognitive"]


def test_facets_partition_the_ontology_into_three() -> None:
    # Factual = objective world, Cognitive = methods,
    # Communication = the speech acts (informs/asks/orders) the agent performs.
    spec = build([_TEXT], KeywordBuilder())
    assert "GDP" in spec.facets["factual"]  # objective-world entities
    assert any(f.startswith("verb:") for f in spec.facets["factual"])  # factual verbs
    assert "method:causal_effect" in spec.facets["cognitive"]  # decision methods
    # Communication facet is exactly the acts realized in speechact.py
    assert set(spec.facets["communication"]) >= {"informs", "asks", "orders"}


def _fake_client(payload: dict[str, object]) -> object:
    text = json.dumps(payload)
    create = lambda **_: SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])  # noqa: E731
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_llm_builder_proposes_subtypes_and_attributes() -> None:
    payload = {
        "entities": [
            {"name": "MacroIndicator"},
            {
                "name": "GDP",
                "subtype_of": "MacroIndicator",
                "attributes": [{"name": "value", "type": "double"}],
            },
            {"name": "CentralBank"},
            {"name": "PolicyRate"},
        ],
        "relations": [["CentralBank", "sets", "PolicyRate"]],
        "verbs": [["RATE_CHANGE", "institutional"]],
        "data_needs": ["GDP", "PolicyRate"],
        "method_needs": ["forecast", "causal_effect"],
    }
    spec = build([_TEXT], LLMBuilder(client=_fake_client(payload)))
    onto = load_ontology_str(spec.ontology_yaml)  # a rich proposal still passes the type system

    assert onto.entities["GDP"].subtype_of == "MacroIndicator"  # subtype
    props = {p.name for p in onto.entities["GDP"].properties}
    assert "value" in props  # typed attribute
    assert any(v.name == "RATE_CHANGE" for v in onto.verbs.values())  # action verb
    assert "causal_effect" in spec.method_needs
