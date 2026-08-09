"""End-to-end walking-skeleton test: one question flows through all five stages."""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def test_answer_walks_end_to_end() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={
            "query_id": "q1",
            "question": "What is the GDP outlook if the CentralBank moves the PolicyLever?",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The chain produced every stage's artifact.
    assert body["query_id"] == "q1"
    assert body["formalized_query"]["query_id"] == "q1"  # 3 grounding -> q*
    assert body["world_model"]["query_id"] == "q1"       # 2 compiler  -> W(q,t)
    assert body["scenarios"], "expected at least one scenario"  # 4 simulation
    assert body["decision"]["query_id"] == "q1"          # 5 policy -> memo
    assert body["decision"]["audit_manifest"], "expected an audit hash"

    # Honesty about what is real vs stub.
    assert body["stages"]["compiler"] == "real"
    assert body["stages"]["grounding"].startswith("real")  # "real (keyword)" or "real (llm)"
    assert body["stages"]["simulation"] in ("basic", "stub")
    assert body["stages"]["policy"] == "basic"


def test_orders_query_applies_causal_method() -> None:
    """A counterfactual ('if ...') routes to orders/METHOD and reads the real causal slice."""
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={
            "query_id": "q3",
            "question": "What happens to GDP if the CentralBank cuts the PolicyLever?",
        },
    )
    assert resp.status_code == 200, resp.text
    retrieval = resp.json()["retrieval"]
    assert retrieval["act"] == "orders"          # counterfactual -> orders
    assert retrieval["kind"] == "method"
    assert retrieval["method"] == "causal_effect"
    # The method read Γ(q): a real causal effect with an identification status came back.
    result = retrieval["result"]
    assert result["answer"] == "causal_effect"
    assert result["effects"], "expected at least one causal effect from Γ(q)"
    assert "identification_status" in result["effects"][0]


def test_asks_query_retrieves_data() -> None:
    """A descriptive question routes to asks/DATA and retrieves the state slice."""
    client = TestClient(create_app())
    resp = client.post("/answer", json={"query_id": "q4", "question": "Give the GDP reading."})
    assert resp.status_code == 200, resp.text
    retrieval = resp.json()["retrieval"]
    assert retrieval["act"] == "asks"
    assert retrieval["kind"] == "data"
    assert "facts" in retrieval


def test_answer_grounds_targets_from_ontology() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={"query_id": "q2", "question": "Give the GDP reading."},
    )
    assert resp.status_code == 200, resp.text
    # GDP is an ontology entity type, so grounding should bind it as a target.
    assert "GDP" in resp.json()["world_model"]["state_package"]["entities"]
