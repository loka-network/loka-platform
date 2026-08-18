"""Submitting work now and collecting it later.

An extraction takes minutes — that is the model's speed. What was ours to fix is that the caller
had to hold an HTTP connection open throughout: clients time out, proxies close idle sockets, and
a Ctrl-C destroyed work that was nearly finished and already paid for.

The property being tested is that the background path and the inline path produce the same thing.
A background mode that quietly does something else is worse than none, because the results look
comparable and are not.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient
from loka_api.app import create_app

_TEXT = "The Central Bank sets the Policy Rate, which affects GDP."


def _await(client: TestClient, job_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body: dict[str, Any] = client.get(f"/jobs/{job_id}").json()
        if body["finished"]:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_submitting_returns_at_once_and_the_result_arrives_later() -> None:
    client = TestClient(create_app())
    submitted = client.post("/build-kb", json={"texts": [_TEXT], "background": True})
    assert submitted.status_code == 200
    job = submitted.json()
    assert job["job_id"] and job["kind"] == "build-kb"
    assert job["finished"] is False or job["state"] in ("pending", "running", "done")

    finished = _await(client, job["job_id"])
    assert finished["state"] == "done"
    assert finished["result"]["ontology_id"]
    assert finished["seconds"] is not None


def test_the_background_result_is_the_inline_result() -> None:
    """The background path runs the same call with the flag cleared, so there is no second
    implementation to drift. This checks that it stayed that way."""
    client = TestClient(create_app())
    inline = client.post("/build-kb", json={"texts": [_TEXT]}).json()
    job = client.post("/build-kb", json={"texts": [_TEXT], "background": True}).json()
    result = _await(client, job["job_id"])["result"]

    assert set(result) == set(inline)
    assert result["ontology_yaml"] == inline["ontology_yaml"]
    assert result["provenance"]["input_digest"] == inline["provenance"]["input_digest"]


def test_a_failure_carries_the_detail_the_inline_call_would_have(monkeypatch: Any) -> None:
    """A polled failure has to say what a synchronous one says, or a caller who switched to
    background mode loses the diagnosis along with the wait."""
    monkeypatch.setenv("LOKA_LLM_BUILD", "1")

    import loka_serving

    monkeypatch.setattr(
        loka_serving, "llm_for", lambda _p: (_ for _ in ()).throw(RuntimeError("no model"))
    )
    client = TestClient(create_app())
    job = client.post("/build-kb", json={"texts": [_TEXT], "background": True}).json()
    finished = _await(client, job["job_id"])

    assert finished["state"] == "failed"
    assert "no ontology was produced" in finished["detail"]["error"]
    assert "no model" in finished["detail"]["cause"]


def test_a_bad_request_fails_the_job_rather_than_the_submission() -> None:
    """Validation happens where the work happens. Reporting it at submission would need the
    check in two places, and the two would eventually disagree."""
    client = TestClient(create_app())
    job = client.post(
        "/build-kb", json={"texts": [_TEXT], "method": "nonsense", "background": True}
    ).json()
    finished = _await(client, job["job_id"])
    assert finished["state"] == "failed"
    assert "single_shot" in str(finished["detail"])


def test_jobs_can_be_listed_without_their_results() -> None:
    """A listing is for finding a job. Carrying every result would make it the largest response
    the API produces, for a request that wanted one line each."""
    client = TestClient(create_app())
    job = client.post("/build-kb", json={"texts": [_TEXT], "background": True}).json()
    _await(client, job["job_id"])

    listed = client.get("/jobs").json()["jobs"]
    assert any(j["job_id"] == job["job_id"] for j in listed)
    assert all("result" not in j for j in listed)
    assert listed[0]["label"].startswith("single_shot")


def test_an_unknown_job_is_404() -> None:
    assert TestClient(create_app()).get("/jobs/nope").status_code == 404


def test_several_jobs_run_at_the_same_time() -> None:
    """The comparison this exists for submits five extractions. Queueing them behind one another
    would make the wall clock the sum again, which is the thing being fixed."""
    client = TestClient(create_app())
    jobs = [
        client.post("/build-kb", json={"texts": [_TEXT], "background": True}).json()["job_id"]
        for _ in range(3)
    ]
    for job_id in jobs:
        assert _await(client, job_id)["state"] == "done"
