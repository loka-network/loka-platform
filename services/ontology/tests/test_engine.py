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
