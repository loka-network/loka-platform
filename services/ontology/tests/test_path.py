"""Multi-hop: the route between entity types is derived from Ω, not hand-written as a join.

A single-entity ontology cannot exercise R or ⪯ — every question is answerable from one row. The
supply ontology declares three relations (each naming the field it is traversed by) and one
subtype, so a question like "which seller shipped this order" has no answer in any single record:
the route has to come from the ontology.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
from loka_ontology import OntologyEngine, load_ontology_str
from loka_ontology.model import Relation

_HERE = os.path.dirname(__file__)
_PATH = os.path.join(_HERE, "..", "..", "..", "examples", "supply_ontology.yaml")


@pytest.fixture
def engine() -> OntologyEngine:
    if not os.path.exists(_PATH):
        pytest.skip("supply ontology not present in this env")
    with open(_PATH) as f:
        return OntologyEngine(load_ontology_str(f.read()))


def _names(path: Sequence[tuple[Relation, bool]]) -> list[str]:
    return [f"{rel.name}{'>' if fwd else '<'}" for rel, fwd in path]


def test_a_two_hop_route_is_derived_from_the_declared_relations(engine: OntologyEngine) -> None:
    # "Which seller shipped this order?" — the answer is in no single record.
    path = engine.path_between("Order", "Seller")
    assert path is not None
    assert _names(path) == ["contains>", "sold_by>"]


def test_relations_are_walked_backwards_for_impact_range(engine: OntologyEngine) -> None:
    # "Tighten the weight spec — which orders are affected?" walks the same relations in reverse.
    assert _names(engine.path_between("BulkyProduct", "Order") or []) == ["contains<"]
    assert _names(engine.path_between("BulkyProduct", "Customer") or []) == [
        "contains<", "placed_by>",
    ]


def test_a_subtype_uses_its_supertype_relations(engine: OntologyEngine) -> None:
    # BulkyProduct declares no relations of its own; ⪯ gives it Product's.
    assert _names(engine.path_between("BulkyProduct", "Seller") or []) == ["sold_by>"]


def test_the_shortest_route_is_returned(engine: OntologyEngine) -> None:
    assert _names(engine.path_between("Product", "Seller") or []) == ["sold_by>"]
    assert len(engine.path_between("Customer", "Seller") or []) == 3


def test_every_step_declares_the_field_it_is_traversed_by(engine: OntologyEngine) -> None:
    # Without `via` a relation is a type-level statement that cannot be followed in data.
    path = engine.path_between("Order", "Seller") or []
    assert all(rel.via for rel, _ in path)
    assert engine.traversable("Order", "Seller") is True


def test_reaching_a_subtype_requires_narrowing_and_is_reported_as_such(
    engine: OntologyEngine,
) -> None:
    # An Order reaches Product; whether that product is bulky is a runtime check, not a guarantee.
    assert engine.path_between("Order", "BulkyProduct") is None
    assert engine.needs_narrowing("Order", "BulkyProduct") is True
    widened = engine.path_between("Order", "BulkyProduct", allow_narrowing=True)
    assert _names(widened or []) == ["contains>"]


def test_a_genuinely_unrelated_pair_is_unreachable(engine: OntologyEngine) -> None:
    # Product -> BulkyProduct is a pure downcast: no relation is involved at all.
    assert engine.path_between("Product", "BulkyProduct") is None
    assert engine.needs_narrowing("Product", "BulkyProduct") is False


def test_an_unknown_type_has_no_route(engine: OntologyEngine) -> None:
    assert engine.path_between("Order", "Warehouse") is None


def test_a_relation_without_via_is_not_traversable() -> None:
    # Ω can state that two types are related without saying how to follow the link.
    onto = load_ontology_str(
        "version: t\nentities:\n  - type: A\n  - type: B\n"
        "relations:\n  - {name: r, from: A, to: B}\n"  # no via
    )
    e = OntologyEngine(onto)
    assert e.path_between("A", "B") is not None   # the type-level route exists
    assert e.traversable("A", "B") is False       # but it cannot be walked in data
