"""Structural inference run against the real marketplace tables, and compared with the ontology
a person wrote for the same domain.

Unit tests show the method behaves as specified on data built to exercise it. This shows what it
does on 4,000 real orders, against a hand-written ontology that was produced independently — the
only check that says whether the method is worth anything.

Both directions of the result are asserted, including the parts where inference and the human
disagree. A comparison that only records agreement measures nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from loka_api.supply import load_supply_dataset, load_supply_ontology
from loka_ontology.compare import compare_ontologies, grounding_checklist
from loka_ontology.infer_tables import infer_ontology_from_tables

_TABLES = ("Order", "OrderItem", "Product", "BulkyProduct", "Seller", "Customer")


@pytest.fixture(scope="module")
def inferred() -> tuple[Any, Any]:
    data = load_supply_dataset()
    if len(data.get("Order", [])) < 100:
        pytest.skip("bundled supply sample not present")
    return infer_ontology_from_tables({name: data[name] for name in _TABLES})


@pytest.fixture(scope="module")
def handwritten() -> Any:
    eng = load_supply_ontology()
    if eng is None:
        pytest.skip("supply ontology not present")
    return eng.ontology


def test_every_relation_the_reviewed_ontology_declares_is_found_in_the_data(
    inferred: tuple[Any, Any], handwritten: Any
) -> None:
    """Four relations, written by hand from domain knowledge, recovered by counting values."""
    onto, _ = inferred
    cmp = compare_ontologies(handwritten, onto)
    found = set(cmp.edges_shared) | set(cmp.edges_reversed)
    assert len(found) == 4
    assert not cmp.edges_only_in_a, "a declared relation the data does not support"


def test_no_relation_is_invented(inferred: tuple[Any, Any], handwritten: Any) -> None:
    """The expensive failure. On six tables with 27 columns, a method prone to false positives
    would find links nobody declared — and each one would be a join the platform is willing to
    walk on evidence that is a coincidence."""
    onto, _ = inferred
    cmp = compare_ontologies(handwritten, onto)
    assert not cmp.edges_only_in_b


def test_one_relation_is_read_from_the_other_end_and_that_is_not_an_error(
    inferred: tuple[Any, Any], handwritten: Any
) -> None:
    """The foreign key sits on OrderItem, so the data says OrderItem -> Order. The reviewed
    ontology reads the same edge as Order --contains--> OrderItem. Both describe one fact; which
    end to read it from is a modelling choice, which is why orientation is listed among the
    things inference does not claim to settle."""
    onto, report = inferred
    cmp = compare_ontologies(handwritten, onto)
    assert cmp.edges_reversed == ("Order -> OrderItem",)
    assert any("orientation" in n for n in report.not_inferable)


def test_the_data_contradicts_a_cardinality_the_reviewer_assumed(
    inferred: tuple[Any, Any], handwritten: Any
) -> None:
    """The reviewed ontology declares Order -> Customer as many-to-one — the natural reading,
    since a customer places many orders. In this dataset it is one-to-one: the publisher issues
    a fresh customer_id per order, and the person is a different column that was not carried in.

    Inference has no domain knowledge to be misled by, so it reports what the values do. This is
    the case that makes cross-checking worth having: a human assumption that the data does not
    support, surfaced as a disagreement rather than shipped as a declaration.
    """
    onto, _ = inferred
    cmp = compare_ontologies(handwritten, onto)
    assert cmp.cardinality_differs == (("Order -> Customer", "many_to_one", "one_to_one"),)


def test_the_subtype_order_is_recovered(inferred: tuple[Any, Any], handwritten: Any) -> None:
    """BulkyProduct ⪯ Product, from nothing but key containment and column overlap."""
    onto, _ = inferred
    cmp = compare_ontologies(handwritten, onto)
    assert cmp.subtypes_shared == ("BulkyProduct <= Product",)
    assert not cmp.subtypes_only_in_a and not cmp.subtypes_only_in_b


def test_nothing_governing_is_produced(inferred: tuple[Any, Any], handwritten: Any) -> None:
    """The boundary, stated as numbers. The reviewed ontology carries verbs, constraints,
    actions and norms; inference produces none of them, and that gap is the reason a data-built
    draft cannot authorise an answer on its own."""
    onto, _ = inferred
    cmp = compare_ontologies(handwritten, onto)
    assert cmp.verbs == (3, 0)
    assert cmp.constraints == (3, 0)
    assert cmp.actions == (4, 0)
    assert cmp.norms == (2, 0)


def test_the_cross_check_turns_disagreement_into_review_items(
    inferred: tuple[Any, Any], handwritten: Any
) -> None:
    """What a reviewer is handed when a text-built ontology meets the data behind it."""
    onto, _ = inferred
    items = grounding_checklist(handwritten, onto)
    kinds = {i["kind"] for i in items}
    assert "cardinality_disagreement" in kinds
    detail = next(i for i in items if i["kind"] == "cardinality_disagreement")["detail"]
    assert "one_to_one" in detail and "many_to_one" in detail


def test_a_concept_with_no_table_behind_it_is_flagged(inferred: tuple[Any, Any]) -> None:
    """The item the two-line design exists to produce. A model reading prose proposes Warehouse;
    no table in the marketplace data supports it. Not wrong — the tables in hand may simply not
    cover warehousing — but it rests on a different kind of evidence from Seller, which every
    row of a real table backs, and the reviewer is told which is which."""
    from loka_ontology import load_ontology_str

    data_inferred, _ = inferred
    text_built = load_ontology_str(
        "version: v1\nentities:\n  - type: Seller\n  - type: Warehouse\n"
    )
    items = grounding_checklist(text_built, data_inferred)
    ungrounded = {i["target"] for i in items if i["kind"] == "ungrounded_entity"}
    assert ungrounded == {"Warehouse"}      # prose only
    assert "Seller" not in ungrounded       # real rows behind it

    # and the reverse direction: five entity types the domain text never mentioned
    undescribed = {i["target"] for i in items if i["kind"] == "undescribed_entity"}
    assert "OrderItem" in undescribed
