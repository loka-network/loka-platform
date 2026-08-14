"""Several sources for one causal edge are pooled, and a disagreement is surfaced, not averaged.

Two studies of the same effect are two pieces of evidence about one claim. Keeping only the last
would drop the others; averaging them silently would hide the case that matters — sources that
disagree by more than sampling error. The answer says which it is.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from lifecycle import publish_built_ontology
from loka_api.app import create_app

_TEXT = ["The Central Bank sets the Policy Rate, which affects GDP."]


def _kb(client: TestClient) -> tuple[str, dict[str, Any]]:
    built = client.post("/build-kb", json={"texts": _TEXT}).json()
    publish_built_ontology(client, built)
    return built["kb_id"], built


def _ask(client: TestClient, kb_id: str) -> dict[str, Any]:
    resp = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": kb_id},
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def _claim(mean: float, se: float, ref: str) -> dict[str, Any]:
    return {
        "cause": "PolicyRate", "effect": "GDP", "mean": mean, "se": se,
        "identification_status": "quasi_experimental", "evidence_refs": [ref],
    }


def test_agreeing_sources_are_pooled_and_reported_consistent() -> None:
    client = TestClient(create_app())
    kb_id, _ = _kb(client)
    client.post(f"/kb/{kb_id}/ingest", json={"causal": [
        _claim(-1.0, 0.2, "paper:a"),
        _claim(-1.1, 0.2, "paper:b"),   # within sampling error of the first
    ]})
    body = _ask(client, kb_id)
    assert body["evidence_conflicts"] == []
    assert body["stages"]["evidence"] == "consistent"


def test_sources_that_disagree_beyond_sampling_error_are_surfaced() -> None:
    client = TestClient(create_app())
    kb_id, _ = _kb(client)
    client.post(f"/kb/{kb_id}/ingest", json={"causal": [
        _claim(-2.0, 0.1, "paper:cut_helps"),
        _claim(+2.0, 0.1, "paper:cut_hurts"),   # opposite sign, tight intervals
    ]})
    body = _ask(client, kb_id)
    assert body["stages"]["evidence"] == "conflicted"
    conflicts = body["evidence_conflicts"]
    assert conflicts, "a sign flip between tight estimates must open a contradiction"
    assert conflicts[0]["claim_id"] == "PolicyRate->GDP"
    # the record names the sources, so a reader can go to them
    assert set(conflicts[0]["between"]) == {"paper:cut_helps", "paper:cut_hurts"}


def test_a_pooled_estimate_reflects_every_source_not_the_last_one() -> None:
    client = TestClient(create_app())
    kb_id, _ = _kb(client)
    client.post(f"/kb/{kb_id}/ingest", json={"causal": [
        _claim(-2.0, 0.2, "paper:a"),
        _claim(-1.0, 0.2, "paper:b"),
    ]})
    body = _ask(client, kb_id)
    claims = body["world_model"]["causal_slice"]["claims"]
    pooled = next(c for c in claims if c["claim_id"] == "PolicyRate->GDP")
    mean = pooled["effect_distribution"]["mean"]
    assert -2.0 < mean < -1.0, f"expected a pooled estimate between the two, got {mean}"
    assert len(pooled["evidence_refs"]) == 2   # both sources are cited


def test_an_answer_with_no_causal_claims_reports_no_evidence_rather_than_consistent() -> None:
    client = TestClient(create_app())
    kb_id, _ = _kb(client)
    body = _ask(client, kb_id)
    assert body["stages"]["evidence"] == "none"   # nothing to agree or disagree about
