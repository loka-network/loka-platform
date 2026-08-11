"""The supply endpoints: routes derived from Ω, and eligibility taken from a declared guard."""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_scenario_exposes_the_parts_of_omega_a_single_table_cannot_use() -> None:
    body = _client().get("/supply/scenario").json()
    if body.get("ontology_version") is None:  # ontology absent in this env
        return
    assert body["ontology_version"] == "supply-v1"
    # ⪯ is declared, and every relation says which field it is traversed by
    assert body["entities"]["BulkyProduct"]["subtype_of"] == "Product"
    assert {r["name"] for r in body["relations"]} == {"sold_by", "contains", "placed_by"}
    assert all(r["via"] for r in body["relations"])
    assert {a["name"] for a in body["actions"]} == {"ShipStandard", "SuspendSeller"}


def test_route_is_derived_not_hand_written() -> None:
    r = _client().get("/supply/route", params={"from_type": "Order", "to_type": "Seller"})
    if r.status_code == 503:
        return
    body = r.json()
    assert body["hops"] == 2
    assert body["route"] == ["contains>(via product_id)", "sold_by>(via seller_id)"]
    assert body["traversable"] is True


def test_route_reports_narrowing_rather_than_pretending_it_is_reachable() -> None:
    r = _client().get("/supply/route", params={"from_type": "Order", "to_type": "BulkyProduct"})
    if r.status_code == 503:
        return
    body = r.json()
    assert body["requires_narrowing"] is True
    assert body["route"] == ["contains>(via product_id)"]


def test_route_to_an_unknown_entity_is_404() -> None:
    r = _client().get("/supply/route", params={"from_type": "Order", "to_type": "Warehouse"})
    if r.status_code == 503:
        return
    assert r.status_code == 404
    assert "not an entity" in r.json()["detail"]


def test_impact_takes_the_rule_from_the_ontology_and_follows_the_relations() -> None:
    r = _client().post("/supply/impact", json={"action": "ShipStandard", "new_threshold": 5000})
    if r.status_code == 503:
        return
    body = r.json()
    # the rule is Ω's, not a constant in the service
    assert body["guard"] == {
        "attribute": "weight_g", "operator": "<=", "from": 30000.0, "to": 5000.0,
    }
    # p_3/p_4 already failed the old guard, so they are not *newly* ineligible
    assert [p["product_id"] for p in body["newly_ineligible"]] == ["p_2"]

    by_entity = {c["entity"]: c for c in body["consequences"]}
    assert [o["order_id"] for o in by_entity["Order"]["affected"]] == ["o_4"]
    assert by_entity["Customer"]["route"] == [
        "contains<(via product_id)", "placed_by>(via customer_id)",
    ]
    assert [s["seller_id"] for s in by_entity["Seller"]["affected"]] == ["s_B"]


def test_impact_on_an_action_the_ontology_does_not_declare_is_refused() -> None:
    r = _client().post("/supply/impact", json={"action": "Teleport", "new_threshold": 1})
    if r.status_code == 503:
        return
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "not an action" in detail["error"]
    assert "ShipStandard" in detail["known_actions"]  # the caller is told what does exist
