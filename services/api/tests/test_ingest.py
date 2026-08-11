"""Build a KB, ingest data + causal, then get real facts and a real causal effect back."""

from __future__ import annotations

from fastapi.testclient import TestClient
from lifecycle import publish_built_ontology
from loka_api.app import create_app


def test_ingest_then_answer_returns_real_data_and_causal() -> None:
    client = TestClient(create_app())

    built = client.post(
        "/build-kb",
        json={"texts": ["The Central Bank sets the Policy Rate, which affects GDP."]},
    ).json()
    kb_id = built["kb_id"]
    publish_built_ontology(client, built)  # a draft ontology cannot authorize an answer

    ing = client.post(
        f"/kb/{kb_id}/ingest",
        json={
            "data": [{"entity": "GDP", "instance": "US", "property": "value", "value": 2.1}],
            "causal": [
                {
                    "cause": "PolicyRate",
                    "effect": "GDP",
                    "mean": -0.8,
                    "se": 0.2,
                    "identification_status": "quasi_experimental",
                    "evidence_refs": ["paper:smith2020"],
                }
            ],
        },
    )
    assert ing.status_code == 200, ing.text
    assert ing.json() == {"kb_id": kb_id, "data_ingested": 1, "causal_ingested": 1}

    # asks/DATA now returns the ingested state value.
    data_ans = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": kb_id},
    ).json()
    facts = data_ans["retrieval"]["facts"]
    assert any(k.startswith("GDP.") for k in facts), facts
    assert data_ans["stages"]["causal"] == "real"  # a causal graph is now attached

    # orders/METHOD returns the ingested causal effect with its identification status.
    causal_ans = client.post(
        "/answer",
        json={
            "query_id": "q2",
            "question": "What happens to GDP if the PolicyRate is cut?",
            "kb_id": kb_id,
        },
    ).json()
    assert causal_ans["retrieval"]["act"] == "orders"
    effects = causal_ans["retrieval"]["result"]["effects"]
    hit = [e for e in effects if e["effect"] == "GDP"]
    assert hit, effects
    assert hit[0]["identification_status"] == "quasi_experimental"
    assert hit[0]["evidence_refs"] == ["paper:smith2020"]


def test_ingest_unknown_kb_is_404() -> None:
    client = TestClient(create_app())
    resp = client.post("/kb/nope/ingest", json={"data": [], "causal": []})
    assert resp.status_code == 404
