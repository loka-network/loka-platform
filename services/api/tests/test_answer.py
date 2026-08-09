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
    assert body["stages"]["grounding"] == "real"
    assert body["stages"]["simulation"] == "stub"
    assert body["stages"]["policy"] == "stub"


def test_answer_grounds_targets_from_ontology() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={"query_id": "q2", "question": "Give the GDP reading."},
    )
    assert resp.status_code == 200, resp.text
    # GDP is an ontology entity type, so grounding should bind it as a target.
    assert "GDP" in resp.json()["world_model"]["state_package"]["entities"]
