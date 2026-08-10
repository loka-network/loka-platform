"""Workflow A over HTTP: POST domain texts -> a validated KB spec."""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def test_build_kb_endpoint() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/build-kb",
        json={
            "texts": [
                "The Central Bank sets the Policy Rate, which affects the Exchange Rate "
                "and GDP. Analysts forecast GDP and estimate the effect of a rate change."
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    spec = resp.json()
    assert spec["ontology_yaml"].startswith("version:")
    assert "GDP" in spec["data_needs"]
    assert "causal_effect" in spec["method_needs"]
    assert spec["facets"]["factual"]
    assert spec["builder"] == "keyword"  # rule-based by default (no LLM configured)


def test_build_kb_rejects_empty() -> None:
    client = TestClient(create_app())
    assert client.post("/build-kb", json={"texts": []}).status_code == 400
