"""Closed loop: build a KB from domain text (Workflow A), then answer against it (Workflow B)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from lifecycle import publish_built_ontology
from loka_api.app import create_app


def test_build_then_answer_against_built_kb() -> None:
    client = TestClient(create_app())

    # Workflow A — build a KB from text; get its id.
    built = client.post(
        "/build-kb",
        json={"texts": ["The Central Bank sets the Policy Rate, which affects GDP."]},
    )
    assert built.status_code == 200, built.text
    kb_id = built.json()["kb_id"]
    assert kb_id
    # The built ontology is a draft: review it through CΩ and publish before it may answer.
    publish_built_ontology(client, built.json())

    # Workflow B — ask a question against THAT built KB.
    ans = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": kb_id},
    )
    assert ans.status_code == 200, ans.text
    body = ans.json()
    # Grounded + compiled against the just-built ontology (GDP is one of its entities).
    assert "GDP" in body["world_model"]["state_package"]["entities"]
    # The built KB has no causal edges yet, so the chain honestly reports causal as empty.
    assert body["stages"]["causal"] == "empty"


def test_answer_with_unknown_kb_id_is_404() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": "does-not-exist"},
    )
    assert resp.status_code == 404
