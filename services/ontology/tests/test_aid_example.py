"""Acceptance test: the aid / grant-allocation ontology is a real, loadable domain.

Guards the first real domain content (examples/aid_allocation.yaml): the engine loads
its entities, relations and constraints, and preserves the provenance annotations that
distinguish real data from model-inferred / exogenous values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loka_ontology import OntologyEngine, load_ontology

AID = Path(__file__).parent.parent / "examples" / "aid_allocation.yaml"


@pytest.fixture
def engine() -> OntologyEngine:
    return OntologyEngine(load_ontology(AID))


def test_all_entities_load(engine: OntologyEngine) -> None:
    assert set(engine.entity_types()) == {
        "Foundation", "Grant", "Program", "Region",
        "BeneficiaryGroup", "Outcome", "ExternalFactor",
    }


def test_relations_present(engine: OntologyEngine) -> None:
    ont = load_ontology(AID)
    names = {r.name for r in ont.relations}
    assert {"grants", "funds", "produces", "influences"} <= names


def test_constraint_only_foundation_approves(engine: OntologyEngine) -> None:
    ont = load_ontology(AID)
    c = next(c for c in ont.constraints if c.verb == "APPROVE")
    assert c.agent_must_be == "Foundation"
    assert c.target_must_be == ("Grant",)


def test_provenance_annotations_preserved(engine: OntologyEngine) -> None:
    # cost_efficiency is a model-inferred value, not real data — the distinction the
    # plan requires must survive loading (kept in the property description for now).
    ont = load_ontology(AID)
    props = {p.name: p for p in ont.entities["Outcome"].properties}
    assert props["cost_efficiency"].description is not None
    assert "inferred" in props["cost_efficiency"].description
    assert "real" in (props["beneficiaries"].description or "")
