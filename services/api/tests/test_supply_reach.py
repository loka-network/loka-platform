"""The cross-table question, asked over HTTP.

The traversal existed and worked, and was reachable only from inside the impact endpoint — so
over HTTP the question could not be asked at all, and the paper said it could not be expressed.
It can: the route comes from the declared verbs and is walked with the fields they declare.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def _reach(**body: object) -> tuple[int, dict]:
    resp = TestClient(create_app()).post("/supply/reach", json=body)
    return resp.status_code, dict(resp.json())


def test_bulky_products_sold_by_sellers_in_one_state() -> None:
    """Two hops, no join written anywhere: the route is derived and walked with declared keys."""
    status, body = _reach(
        from_type="Seller", to_type="BulkyProduct", where={"seller_state": "SP"}, limit=3
    )
    assert status == 200, body
    assert body["from"]["matched"] > 0
    assert body["route"] == [
        "fulfilled_by<(via seller_id)",
        "of_product>(via product_id)",
    ]
    assert body["reached"] > 0
    assert len(body["rows"]) == 3 and body["truncated"]


def test_a_filter_on_an_undeclared_attribute_is_refused() -> None:
    """Not silently matched against nothing. An empty result reads as an answer — no sellers
    qualify — where the truth is that the question could not be asked."""
    status, body = _reach(
        from_type="Seller", to_type="BulkyProduct", where={"credit_score": "x"}
    )
    assert status == 400
    assert "credit_score" in body["detail"]
    assert "supply-v2" in body["detail"]


def test_an_entity_the_ontology_does_not_have_is_refused() -> None:
    status, body = _reach(from_type="Warehouse", to_type="Seller")
    assert status == 404
    assert "Warehouse" in body["detail"]


def test_a_pair_with_no_declared_route_says_so() -> None:
    """A missing route is reported as a missing route, not as zero rows."""
    status, body = _reach(from_type="Customer", to_type="BulkyProduct")
    assert status == 200
    if body["route"] is None:
        assert body["reason"]
        assert body["reached"] == 0
