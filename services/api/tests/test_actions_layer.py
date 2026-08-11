"""The action layer proposes a governed action from the ontology, gated and awaiting confirmation."""

from __future__ import annotations

from fastapi.testclient import TestClient
from lifecycle import publish_built_ontology
from loka_api.app import create_app


def test_answer_proposes_governed_action() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/answer",
        json={
            "query_id": "q1",
            "question": "What if the CentralBank moves the PolicyLever, and GDP?",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stages"]["action"] == "basic"

    actions = body["actions"]
    assert actions, "expected at least one proposed action"
    cut = next(a for a in actions if a["action_name"] == "CutPolicyLever")
    assert cut["verb"] == "RATE_CHANGE"
    assert cut["target"] == "PolicyLever"
    assert cut["requires_confirmation"] is True  # governed: nothing auto-executes
    # guard "policy_rate > 0" evaluated against state (CentralBank.Fed.policy_rate = 0.0525)
    assert cut["guard_status"] == "satisfied"
    assert cut["status"] == "proposed"


def test_built_kb_without_actions_has_none() -> None:
    client = TestClient(create_app())
    built = client.post(
        "/build-kb", json={"texts": ["The Central Bank sets the Policy Rate which affects GDP."]}
    ).json()
    kb_id = built["kb_id"]
    publish_built_ontology(client, built)  # a draft ontology cannot authorize an answer
    body = client.post(
        "/answer", json={"query_id": "q2", "question": "Give the GDP reading.", "kb_id": kb_id}
    ).json()
    # a text-built ontology has no action types yet -> no proposals, reported honestly
    assert body["actions"] == []
    assert body["stages"]["action"] == "none"
