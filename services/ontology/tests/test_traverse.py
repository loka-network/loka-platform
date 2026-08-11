"""Walking an Ω-derived route over data: the answer, not just the direction."""

from __future__ import annotations

import os

import pytest
from loka_ontology import OntologyEngine, load_ontology_str
from loka_ontology.loader import OntologyLoadError
from loka_ontology.traverse import (
    Dataset,
    TraversalError,
    follow,
    reach,
    rows_of_type,
)

_HERE = os.path.dirname(__file__)
_PATH = os.path.join(_HERE, "..", "..", "..", "examples", "supply_ontology.yaml")

_DATA: Dataset = {
    "Seller": [
        {"seller_id": "s_A", "seller_state": "SP", "on_time_rate": 0.91},
        {"seller_id": "s_B", "seller_state": "RJ", "on_time_rate": 0.74},
    ],
    "Product": [
        {"product_id": "p_1", "weight_g": 1200},
        {"product_id": "p_2", "weight_g": 8000},
    ],
    "BulkyProduct": [
        {"product_id": "p_3", "weight_g": 45000},
        {"product_id": "p_4", "weight_g": 32000},
    ],
    # The junction: the seller is a property of the order line, not of the product.
    "OrderItem": [
        {"item_id": "i_1", "order_id": "o_1", "product_id": "p_1", "seller_id": "s_A"},
        {"item_id": "i_2", "order_id": "o_2", "product_id": "p_3", "seller_id": "s_A"},
        {"item_id": "i_3", "order_id": "o_3", "product_id": "p_4", "seller_id": "s_B"},
        {"item_id": "i_4", "order_id": "o_4", "product_id": "p_2", "seller_id": "s_B"},
    ],
    "Order": [
        {"order_id": "o_1", "customer_id": "c_X", "days_late": -2.0},
        {"order_id": "o_2", "customer_id": "c_Y", "days_late": 6.0},
        {"order_id": "o_3", "customer_id": "c_Z", "days_late": 11.0},
        {"order_id": "o_4", "customer_id": "c_X", "days_late": 1.0},
    ],
    "Customer": [
        {"customer_id": "c_X", "customer_state": "RJ"},
        {"customer_id": "c_Y", "customer_state": "SP"},
        {"customer_id": "c_Z", "customer_state": "MG"},
    ],
}


@pytest.fixture
def engine() -> OntologyEngine:
    if not os.path.exists(_PATH):
        pytest.skip("supply ontology not present in this env")
    with open(_PATH) as f:
        return OntologyEngine(load_ontology_str(f.read()))


def test_two_hops_answer_a_question_no_single_row_contains(engine: OntologyEngine) -> None:
    # "Which seller shipped order o_2?" — Order has no seller field.
    o2 = [r for r in _DATA["Order"] if r["order_id"] == "o_2"]
    out = reach(engine, _DATA, from_type="Order", to_type="Seller", start=o2)
    assert out["hops"] == 2
    assert out["route"] == ["contains>(via order_id)", "fulfilled_by>(via seller_id)"]
    assert [r["seller_id"] for r in out["rows"]] == ["s_A"]


def test_impact_range_walks_the_relations_backwards(engine: OntologyEngine) -> None:
    # "Tighten the standard-shipping limit to 5kg — what does it touch?"
    over = [p for p in rows_of_type(engine, _DATA, "Product") if p["weight_g"] > 5000]
    assert {p["product_id"] for p in over} == {"p_2", "p_3", "p_4"}

    orders = reach(engine, _DATA, from_type="Product", to_type="Order", start=over)
    assert {r["order_id"] for r in orders["rows"]} == {"o_2", "o_3", "o_4"}

    customers = reach(engine, _DATA, from_type="Product", to_type="Customer", start=over)
    assert {r["customer_id"] for r in customers["rows"]} == {"c_X", "c_Y", "c_Z"}
    assert customers["hops"] == 3  # Product <- OrderItem <- Order -> Customer

    sellers = reach(engine, _DATA, from_type="Product", to_type="Seller", start=over)
    assert {r["seller_id"] for r in sellers["rows"]} == {"s_A", "s_B"}


def test_subtype_rows_are_visible_when_reading_the_supertype(engine: OntologyEngine) -> None:
    # A BulkyProduct is a Product, so it appears when Products are read.
    ids = {p["product_id"] for p in rows_of_type(engine, _DATA, "Product")}
    assert ids == {"p_1", "p_2", "p_3", "p_4"}
    assert {p["product_id"] for p in rows_of_type(engine, _DATA, "BulkyProduct")} == {"p_3", "p_4"}


def test_narrowing_is_applied_when_the_target_is_a_subtype(engine: OntologyEngine) -> None:
    # "Which orders contain a bulky product?" reaches Product, then narrows.
    out = reach(engine, _DATA, from_type="Order", to_type="BulkyProduct", start=_DATA["Order"])
    assert out["requires_narrowing"] is True
    assert {p["product_id"] for p in out["rows"]} == {"p_3", "p_4"}  # p_1/p_2 dropped


def test_a_route_the_ontology_does_not_declare_is_reported_not_guessed(
    engine: OntologyEngine,
) -> None:
    out = reach(engine, _DATA, from_type="Customer", to_type="Warehouse", start=[])
    assert out["route"] is None
    assert "declares no route" in out["reason"]


def test_following_a_relation_without_via_raises_instead_of_guessing_a_key() -> None:
    onto = load_ontology_str(
        "version: t\nentities:\n"
        "  - type: A\n    properties:\n      - {name: b_id, type: string}\n"
        "  - type: B\n    properties:\n      - {name: b_id, type: string}\n"
        "relations:\n  - {name: r, from: A, to: B}\n"  # no via
    )
    e = OntologyEngine(onto)
    path = e.path_between("A", "B")
    assert path is not None
    with pytest.raises(TraversalError, match="no 'via'"):
        data: Dataset = {"A": [{"b_id": "1"}], "B": [{"b_id": "1"}]}
        follow(e, data, path, [{"b_id": "1"}], start_type="A")


def test_c_omega_rejects_a_via_field_missing_on_either_side() -> None:
    # R8: a link field must exist on both types, or the route could not be walked.
    with pytest.raises(OntologyLoadError, match="does not declare that property"):
        load_ontology_str(
            "version: t\nentities:\n"
            "  - type: A\n    properties:\n      - {name: b_id, type: string}\n"
            "  - type: B\n"  # B never declares b_id
            "relations:\n  - {name: r, from: A, to: B, via: b_id}\n"
        )


def test_a_subtype_inherits_the_link_field_for_r8() -> None:
    # BulkyProduct declares no seller_id of its own; it inherits Product's, so sold_by holds.
    load_ontology_str(
        "version: t\nentities:\n"
        "  - type: P\n    properties:\n      - {name: s_id, type: string}\n"
        "  - type: BP\n    subtype_of: P\n"
        "  - type: S\n    properties:\n      - {name: s_id, type: string}\n"
        "relations:\n  - {name: sold_by, from: BP, to: S, via: s_id}\n"
    )
