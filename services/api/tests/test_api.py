"""Tests for the Loka Platform API (via FastAPI TestClient — no server needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from loka_api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ontology_version"] == "demo-v1"


def test_compile_returns_wqt(client: TestClient) -> None:
    resp = client.post(
        "/compile",
        json={"query_id": "q1", "task_type": "counterfactual", "targets": ["GDP"]},
    )
    assert resp.status_code == 200
    wqt = resp.json()
    assert wqt["query_id"] == "q1"
    # causal slice filled from Γ; state slice pulled from Eₜ
    claim_ids = {c["claim_id"] for c in wqt["causal_slice"]["claims"]}
    assert claim_ids == {"c1", "c2", "c3"}
    assert wqt["state_package"]["state_slice"]["GDP.TH.value"] == 2.1
    assert wqt["manifest"]["omega_version"] == "demo-v1"


def test_compile_unknown_target_is_400(client: TestClient) -> None:
    resp = client.post(
        "/compile",
        json={"query_id": "q2", "task_type": "x", "targets": ["NotAnEntity"]},
    )
    assert resp.status_code == 400
    assert "not in ontology" in resp.json()["detail"]


def test_compile_validates_request_body(client: TestClient) -> None:
    resp = client.post("/compile", json={"query_id": "q3"})  # missing required fields
    assert resp.status_code == 422  # FastAPI request validation
