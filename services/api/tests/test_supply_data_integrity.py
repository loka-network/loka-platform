"""The bundled supply sample must satisfy what Ω declares about it.

Ω says BulkyProduct ⪯ Product and that four relations are walked by named fields. Data that
contradicts those declarations does not fail loudly — it produces a traversal that dead-ends and
a query that quietly returns a biased subset. Both look like ordinary empty results.

These are the checks that caught it: the sample was built by *partitioning* products into plain
and bulky, so ⪯ was inverted (no BulkyProduct was a Product) and 4.7% of order lines pointed at
a product that was not in Product.csv — precisely the heavy ones the ShipStandard guard exists
to catch. The demo of the guard was running on data with the guarded case removed.
"""

from __future__ import annotations

import pytest
from loka_api.supply import load_supply_dataset
from loka_ontology import OntologyEngine


@pytest.fixture(scope="module")
def data() -> dict[str, list[dict[str, object]]]:
    rows = load_supply_dataset()
    if len(rows.get("Order", [])) < 100:
        pytest.skip("bundled supply sample not present")
    return rows


@pytest.fixture(scope="module")
def engine() -> OntologyEngine:
    from loka_api.supply import load_supply_ontology

    eng = load_supply_ontology()
    if eng is None:
        pytest.skip("supply ontology not present")
    return eng


def test_every_subtype_instance_is_an_instance_of_its_supertype(
    data: dict[str, list[dict[str, object]]], engine: OntologyEngine
) -> None:
    """⪯ is containment, not partition. If BulkyProduct ⪯ Product then every bulky product id
    must also appear in Product — otherwise ``of_product`` dead-ends on exactly those rows."""
    assert "Product" in engine.subtypes_of("Product") + engine.supertypes("BulkyProduct")
    products = {r["product_id"] for r in data["Product"]}
    bulky = {r["product_id"] for r in data["BulkyProduct"]}
    assert bulky, "the sample carries no BulkyProduct rows, so ⪯ is untested"
    assert bulky <= products, (
        f"{len(bulky - products)} BulkyProduct rows are not Products; ⪯ is declared as "
        "containment and the data contradicts it"
    )


def test_the_subtype_is_exactly_what_the_guard_threshold_says(
    data: dict[str, list[dict[str, object]]], engine: OntologyEngine
) -> None:
    """The subtype boundary and the action's eligibility rule are one rule, read from Ω. If they
    could drift, a product could be bulky for classification and not bulky for shipping."""
    guard = next(a.guard for a in engine.action_types() if a.name == "ShipStandard")
    threshold = float(guard.rsplit("<=", 1)[1])
    bulky = {r["product_id"] for r in data["BulkyProduct"]}
    over = {
        r["product_id"] for r in data["Product"]
        if isinstance(r.get("weight_g"), float) and r["weight_g"] > threshold
    }
    assert bulky == over


def test_every_declared_relation_resolves_in_the_sample(
    data: dict[str, list[dict[str, object]]], engine: OntologyEngine
) -> None:
    """Each relation names the field it is walked by. A sample where that field points at rows
    it does not contain makes multi-hop traversal fail while looking like missing data.

    Which side to check is decided by the cardinality, not by the direction the relation is
    written in. The key lives on the "many" side: for Order --contains--> OrderItem the field is
    on OrderItem, so what must hold is that every line belongs to an order — not that every
    order has a line. Twenty-six orders in this sample have none, which is true of the source
    data (cancelled and unavailable orders keep no lines) and is not a closure failure.
    """
    checked = 0
    for rel in engine.relations():
        via = rel.via
        if not via:
            continue
        if rel.effective_cardinality.value == "one_to_many":
            holder, referenced = rel.to_type, rel.from_type
        else:
            holder, referenced = rel.from_type, rel.to_type
        if holder not in data or referenced not in data:
            continue
        available = {r[via] for r in data[referenced] if via in r}
        dangling = {r[via] for r in data[holder] if r.get(via) not in available}
        assert not dangling, (
            f"relation {rel.name}: {len(dangling)} {holder}.{via} values have no matching "
            f"{referenced}; the sample is not referentially closed"
        )
        checked += 1
    assert checked == 4, f"only {checked} relations were checked; the rest were skipped"


def test_the_sample_is_large_enough_to_exercise_the_guard(
    data: dict[str, list[dict[str, object]]],
) -> None:
    """A sample that happens to contain no heavy product would pass every check above and
    demonstrate nothing about the guard."""
    assert len(data["BulkyProduct"]) >= 50
    assert len(data["Product"]) >= 1000


def test_the_norms_actually_bite_on_the_real_data(
    data: dict[str, list[dict[str, object]]], engine: OntologyEngine
) -> None:
    """A norm that never fires on the shipped data demonstrates nothing, and would have hidden
    exactly the defect R11 was written for. Both norms are checked against real rows.

    The suspension case is the one worth reading twice. The guard is about the mechanism — is
    this seller's on-time rate below the threshold — and it is correct. But most of the sellers
    it catches are caught on a handful of orders, where the rate is mostly noise. Whether it is
    legitimate to act on that much evidence is a separate question from whether the number is
    below the line, and it is the norm, not the guard, that answers it.
    """
    threshold = float(
        next(a.guard for a in engine.action_types() if a.name == "SuspendSeller").rsplit("<", 1)[1]
    )
    minimum = float(
        next(
            n.when for n in engine.norms_for("SuspendSeller")
            if n.name == "NoSuspensionOnThinEvidence"
        ).rsplit("<", 1)[1]
    )
    caught = [
        r for r in data["Seller"]
        if isinstance(r.get("on_time_rate"), float) and r["on_time_rate"] < threshold
    ]
    withheld = [r for r in caught if float(r.get("delivered_lines") or 0) < minimum]
    assert caught, "no seller is below the guard threshold; the guard is untested"
    assert withheld, "no suspension is withheld for thin evidence; the norm is untested"
    assert len(withheld) / len(caught) > 0.5  # the majority, not a rounding error

    late = [
        r for r in data["Order"]
        if isinstance(r.get("days_late"), float) and r["days_late"] >= 3
    ]
    assert late, "no order is late enough to trigger the disclosure obligation"
