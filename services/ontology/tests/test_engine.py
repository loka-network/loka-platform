"""Acceptance tests: the ontology engine skeleton works.

Demo goal: load the toy ontology → answer subtype questions → check type bindings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loka_ontology import OntologyEngine, load_ontology
from loka_ontology.loader import OntologyLoadError, load_ontology_str

TOY = Path(__file__).parent.parent / "examples" / "toy_ontology.yaml"


@pytest.fixture
def engine() -> OntologyEngine:
    return OntologyEngine(load_ontology(TOY))


# ---- subtype ⪯ ----

def test_subtype_chain_true(engine: OntologyEngine) -> None:
    # SovereignBond ⪯ Bond ⪯ Instrument
    assert engine.is_subtype("SovereignBond", "Instrument") is True
    assert engine.is_subtype("SovereignBond", "Bond") is True


def test_subtype_reflexive(engine: OntologyEngine) -> None:
    assert engine.is_subtype("Bond", "Bond") is True


def test_subtype_false(engine: OntologyEngine) -> None:
    assert engine.is_subtype("Regulator", "Instrument") is False
    assert engine.is_subtype("Instrument", "SovereignBond") is False  # wrong direction


def test_supertypes(engine: OntologyEngine) -> None:
    assert engine.supertypes("SovereignBond") == ["Bond", "Instrument"]


# ---- entity / relation queries ----

def test_entity_types(engine: OntologyEngine) -> None:
    assert set(engine.entity_types()) >= {"Instrument", "Bond", "SovereignBond", "Regulator"}


def test_subtypes_of(engine: OntologyEngine) -> None:
    assert set(engine.subtypes_of("Instrument")) == {"Bond", "SovereignBond"}
    assert engine.subtypes_of("SovereignBond") == []  # leaf


def test_relations_from_uses_subtyping(engine: OntologyEngine) -> None:
    # regulator-of has source Regulator; CentralBank ⪯ Regulator → should match
    names = [r.name for r in engine.relations_from("CentralBank")]
    assert "regulator-of" in names


def test_relations_to_uses_subtyping(engine: OntologyEngine) -> None:
    # regulator-of has target Instrument; SovereignBond ⪯ Instrument → should match
    names = [r.name for r in engine.relations_to("SovereignBond")]
    assert "regulator-of" in names


# ---- verbs ----

def test_verb_class(engine: OntologyEngine) -> None:
    from loka_ontology import VerbClass

    assert engine.verb_class("REGULATE") == VerbClass.INSTITUTIONAL
    assert engine.verb_class("TRADE") == VerbClass.FACTUAL
    assert engine.verb_class("NONEXISTENT") is None


def test_verbs_of_class(engine: OntologyEngine) -> None:
    from loka_ontology import VerbClass

    assert engine.verbs_of_class(VerbClass.INSTITUTIONAL) == ["REGULATE"]
    assert engine.verbs_of_class(VerbClass.FACTUAL) == ["TRADE"]


# ---- typing-constraint checks (CΩ, early form) ----

def test_binding_ok(engine: OntologyEngine) -> None:
    # CentralBank ⪯ Regulator, SovereignBond ⪯ Instrument → legal
    res = engine.check_binding("REGULATE", "CentralBank", "SovereignBond")
    assert res.ok is True
    assert res.rule is not None


def test_binding_agent_violation(engine: OntologyEngine) -> None:
    # Bond is not a Regulator → illegal (type_violation)
    res = engine.check_binding("REGULATE", "Bond", "Instrument")
    assert res.ok is False
    assert res.reason is not None and "type_violation" in res.reason


def test_binding_unknown_verb(engine: OntologyEngine) -> None:
    res = engine.check_binding("FLY", "Regulator", "Instrument")
    assert res.ok is False
    assert res.reason is not None and "undefined verb" in res.reason


# ---- typed properties (⪯-inherited) + value validation ----

def test_properties_are_inherited(engine: OntologyEngine) -> None:
    props = engine.properties_of("SovereignBond")
    # own (issuer_country) + Bond (coupon, maturity) + Instrument (id, notional)
    assert set(props) == {"id", "notional", "coupon", "maturity", "issuer_country"}


def test_property_of(engine: OntologyEngine) -> None:
    from loka_ontology import BaseType

    p = engine.property_of("CentralBank", "policy_rate")
    assert p is not None and p.base_type == BaseType.DOUBLE
    assert engine.property_of("CentralBank", "nope") is None


def test_validate_values_ok(engine: OntologyEngine) -> None:
    from datetime import date

    values = {"id": "US10Y", "notional": 1000.0, "coupon": 2.5, "maturity": date(2030, 1, 1),
              "issuer_country": "US"}
    assert engine.validate_values("SovereignBond", values) == ()


def test_validate_missing_required(engine: OntologyEngine) -> None:
    # id (from Instrument) and issuer_country (own) are required
    errors = engine.validate_values("SovereignBond", {"notional": 1.0})
    assert any("id" in e for e in errors)
    assert any("issuer_country" in e for e in errors)


def test_validate_wrong_type(engine: OntologyEngine) -> None:
    errors = engine.validate_values("Instrument", {"id": "X", "notional": "not-a-number"})
    assert any("notional" in e and "double" in e for e in errors)


def test_validate_bool_is_not_integer_or_double(engine: OntologyEngine) -> None:
    # bool must not satisfy double (it is an int subclass in Python)
    errors = engine.validate_values("Instrument", {"id": "X", "notional": True})
    assert any("notional" in e for e in errors)


def test_validate_unknown_property_optional(engine: OntologyEngine) -> None:
    values = {"id": "X", "extra": 1}
    assert engine.validate_values("Instrument", values) == ()  # allowed by default
    errors = engine.validate_values("Instrument", values, allow_unknown=False)
    assert any("unknown property: extra" in e for e in errors)


def test_loader_rejects_bad_property_type() -> None:
    bad = "version: v0\nentities:\n  - {type: X, properties: [{name: a, type: nope}]}\n"
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)


# ---- link cardinality ----

def _rel_engine(card: str) -> OntologyEngine:
    y = (
        "version: v0\nentities:\n  - {type: A}\n  - {type: B}\n"
        f"relations:\n  - {{name: r, from: A, to: B, cardinality: {card}}}\n"
    )
    return OntologyEngine(load_ontology_str(y))


def test_relation_lookup(engine: OntologyEngine) -> None:
    from loka_ontology import Cardinality

    rel = engine.relation("regulator-of")
    assert rel is not None and rel.cardinality == Cardinality.ONE_TO_MANY
    assert engine.relation("nope") is None


def test_one_to_many_ok_and_violation(engine: OntologyEngine) -> None:
    # regulator-of is one_to_many: one Regulator → many Instruments; each Instrument ≤ 1 Regulator
    assert engine.validate_links("regulator-of", [("Fed", "US10Y"), ("Fed", "AAPL")]) == ()
    errors = engine.validate_links("regulator-of", [("Fed", "US10Y"), ("ECB", "US10Y")])
    assert any("US10Y" in e and "sources" in e for e in errors)


def test_cardinality_dedupes_identical_links(engine: OntologyEngine) -> None:
    assert engine.validate_links("regulator-of", [("Fed", "US10Y"), ("Fed", "US10Y")]) == ()


def test_many_to_many_unconstrained() -> None:
    e = _rel_engine("many_to_many")
    assert e.validate_links("r", [("a1", "b1"), ("a1", "b2"), ("a2", "b1")]) == ()


def test_one_to_one() -> None:
    e = _rel_engine("one_to_one")
    assert e.validate_links("r", [("a1", "b1"), ("a2", "b2")]) == ()
    assert e.validate_links("r", [("a1", "b1"), ("a1", "b2")])  # a1 → 2 targets
    assert e.validate_links("r", [("a1", "b1"), ("a2", "b1")])  # b1 ← 2 sources


def test_many_to_one() -> None:
    e = _rel_engine("many_to_one")
    assert e.validate_links("r", [("a1", "b1"), ("a2", "b1")]) == ()  # many A → one B ok
    assert e.validate_links("r", [("a1", "b1"), ("a1", "b2")])  # a1 → 2 targets: violation


def test_validate_links_unknown_relation(engine: OntologyEngine) -> None:
    assert engine.validate_links("nope", [("a", "b")]) == ("unknown relation: nope",)


def test_loader_rejects_bad_cardinality() -> None:
    bad = (
        "version: v0\nentities:\n  - {type: A}\n  - {type: B}\n"
        "relations:\n  - {name: r, from: A, to: B, cardinality: nope}\n"
    )
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)


# ---- loader structural checks ----

def test_loader_rejects_dangling_subtype() -> None:
    bad = "version: v0\nentities:\n  - {type: Bond, subtype_of: DoesNotExist}\n"
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)


def test_loader_rejects_cycle() -> None:
    bad = (
        "version: v0\nentities:\n"
        "  - {type: A, subtype_of: B}\n"
        "  - {type: B, subtype_of: A}\n"
    )
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)
