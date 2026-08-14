"""Structural inference across tables: keys, links, cardinality, and the subtype order.

The claim being tested is not "it finds relations" but "what it proposes follows from the values,
and what it cannot see it declines to guess". Both halves matter: a method that occasionally
invents a link is worse than one that finds fewer, because the whole reason to read structure out
of data rather than ask a model is that the answer can be checked.
"""

from __future__ import annotations

from loka_ontology.infer_tables import (
    MIN_DISTINCT,
    infer_ontology_from_tables,
)

# Deliberately above MIN_DISTINCT: with fewer distinct values than that, containment stops
# being evidence and the link is correctly refused — as a separate test shows.
_ORDERS = [
    {"order_id": f"o{i}", "customer_id": f"c{i % 6}", "status": "delivered"}
    for i in range(12)
]
_CUSTOMERS = [{"customer_id": f"c{i}", "customer_state": "SP"} for i in range(6)]
_ITEMS = [
    {"item_id": f"i{i}", "order_id": f"o{i % 12}", "price": 10.0 + i}
    for i in range(20)
]


def test_a_foreign_key_becomes_a_relation_carrying_the_field_it_is_walked_by() -> None:
    onto, report = infer_ontology_from_tables({"Order": _ORDERS, "Customer": _CUSTOMERS})
    rel = next(r for r in onto.relations if r.from_type == "Order")
    assert rel.to_type == "Customer"
    assert rel.via == "customer_id"  # without this the relation is declared but not walkable
    link = next(x for x in report.links if x.accepted)
    assert link.matched == link.total == 12
    assert link.reason == "12/12 values are Customer.customer_id"


def test_cardinality_is_counted_not_assumed() -> None:
    """Four customers across twelve orders: several orders point at the same customer. The same
    shape with distinct values is one-to-one. Nothing here is a default."""
    onto, _ = infer_ontology_from_tables({"Order": _ORDERS, "Customer": _CUSTOMERS})
    assert next(r for r in onto.relations).effective_cardinality.value == "many_to_one"

    unique = [dict(o, customer_id=f"c{i}") for i, o in enumerate(_ORDERS)]
    customers = [{"customer_id": f"c{i}", "customer_state": "SP"} for i in range(12)]
    onto2, _ = infer_ontology_from_tables({"Order": unique, "Customer": customers})
    assert next(r for r in onto2.relations).effective_cardinality.value == "one_to_one"


def test_a_shared_vocabulary_is_not_mistaken_for_a_reference() -> None:
    """The failure this method is most prone to. Two tables carry Brazilian state codes in an
    ordinary column; the values overlap completely and neither references the other. Only a
    primary key is a valid target, so neither column is proposed."""
    sellers = [{"seller_id": f"s{i}", "state": "SP" if i % 2 else "RJ"} for i in range(10)]
    customers = [{"customer_id": f"c{i}", "state": "SP" if i % 3 else "RJ"} for i in range(10)]
    onto, report = infer_ontology_from_tables({"Seller": sellers, "Customer": customers})
    assert onto.relations == []
    assert not [x for x in report.links if x.accepted]


def test_a_low_cardinality_column_is_not_evidence_even_when_contained() -> None:
    """Containment in a handful of values is arithmetic. A status column whose three values all
    happen to be keys of a lookup table would otherwise read as a foreign key."""
    codes = [{"status": s, "label": s.title()} for s in ("new", "paid", "shipped")]
    orders = [{"order_id": f"o{i}", "status": "paid"} for i in range(20)]
    _, report = infer_ontology_from_tables({"Order": orders, "StatusCode": codes})
    accepted = [x for x in report.links if x.accepted]
    assert not accepted
    assert MIN_DISTINCT > 1  # the threshold exists; a value of 1 would make this vacuous


def test_a_column_that_resolves_but_is_unrelated_by_name_is_rejected_with_the_counts() -> None:
    """Values resolving is necessary, not sufficient. The rejection keeps the numbers, so a
    reviewer who disagrees can see exactly what was found and overrule it."""
    people = [{"person_id": f"p{i}", "note": ""} for i in range(10)]
    audits = [{"audit_id": f"a{i}", "touched_by": f"p{i}"} for i in range(10)]
    _, report = infer_ontology_from_tables({"Audit": audits, "Person": people})
    rejected = [x for x in report.links if not x.accepted and x.from_column == "touched_by"]
    assert rejected, "a fully-resolving candidate must be reported, not silently dropped"
    assert rejected[0].inclusion == 1.0
    assert rejected[0].name_match == "unrelated"


# ---- ⪯ ----

def test_a_table_whose_keys_and_columns_sit_inside_another_is_its_subtype() -> None:
    products = [{"product_id": f"p{i}", "weight_g": 100.0 * i} for i in range(20)]
    heavy = [r for r in products if r["weight_g"] > 1500]
    onto, report = infer_ontology_from_tables({"Product": products, "Heavy": heavy})
    assert onto.entities["Heavy"].subtype_of == "Product"
    accepted = [s for s in report.subtypes if s.accepted]
    assert len(accepted) == 1
    assert accepted[0].inclusion == 1.0


def test_the_subtype_direction_is_decided_by_size_not_by_order() -> None:
    products = [{"product_id": f"p{i}", "weight_g": 100.0 * i} for i in range(20)]
    heavy = [r for r in products if r["weight_g"] > 1500]
    _, report = infer_ontology_from_tables({"Heavy": heavy, "Product": products})
    wrong = next(s for s in report.subtypes if s.subtype == "Product")
    assert wrong.accepted is False
    assert "not smaller" in wrong.reason


def test_a_table_sharing_a_key_but_adding_columns_is_not_a_subtype() -> None:
    """An order and its shipment share an id without one being a kind of the other. The extra
    columns are named in the rejection so a reviewer sees why."""
    orders = [{"order_id": f"o{i}", "status": "paid"} for i in range(20)]
    shipments = [{"order_id": f"o{i}", "status": "paid", "carrier": "X"} for i in range(10)]
    onto, report = infer_ontology_from_tables({"Order": orders, "Shipment": shipments})
    assert onto.entities["Shipment"].subtype_of is None
    rejected = next(s for s in report.subtypes if s.subtype == "Shipment")
    assert rejected.accepted is False
    assert rejected.extra_columns == ("carrier",)


def test_a_subtype_is_not_also_reported_as_a_relation() -> None:
    """One fact, one reading. A subtype's key is inside its supertype's by definition, so the
    same pair would otherwise appear twice — once as ⪯ and once as a foreign key."""
    products = [{"product_id": f"p{i}", "weight_g": 100.0 * i} for i in range(20)]
    heavy = [r for r in products if r["weight_g"] > 1500]
    onto, _ = infer_ontology_from_tables({"Product": products, "Heavy": heavy})
    assert not [r for r in onto.relations if r.from_type == "Heavy" and r.to_type == "Product"]


# ---- what it refuses to guess ----

def test_nothing_that_is_a_decision_rather_than_an_observation_is_produced() -> None:
    onto, report = infer_ontology_from_tables(
        {"Order": _ORDERS, "Customer": _CUSTOMERS, "OrderItem": _ITEMS}
    )
    assert onto.verbs == {}
    assert onto.constraints == []
    assert onto.actions == []
    assert onto.norms == []
    # and the draft says so, rather than leaving the absence to be noticed later
    assert any("verbs" in n for n in report.not_inferable)
    assert any("norms" in n for n in report.not_inferable)


def test_unrelated_candidates_are_counted_rather_than_listed() -> None:
    """Every column is tried against every key, so most pairs resolve for nothing. Listing them
    would bury the rejections worth reading; dropping them silently would hide how much was
    tried. They are counted."""
    _, report = infer_ontology_from_tables(
        {"Order": _ORDERS, "Customer": _CUSTOMERS, "OrderItem": _ITEMS}
    )
    assert report.unrelated_pairs > 0
    assert all(x.inclusion >= 0.5 for x in report.links)


def test_the_result_round_trips_through_the_loader() -> None:
    """An inferred ontology has to survive CΩ, or it is a proposal the platform cannot load."""
    from loka_ontology import load_ontology_str
    from loka_ontology.infer import to_yaml

    onto, _ = infer_ontology_from_tables(
        {"Order": _ORDERS, "Customer": _CUSTOMERS, "OrderItem": _ITEMS}
    )
    reloaded = load_ontology_str(to_yaml(onto))
    assert set(reloaded.entities) == {"Order", "Customer", "OrderItem"}
    assert all(r.via for r in reloaded.relations)  # R8: the traversal key survived
