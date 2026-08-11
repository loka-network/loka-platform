"""A generated ontology is a proposal: draft → validated → published, with a human in the middle."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from loka_api.app import create_app
from loka_api.ontology_store import review_checklist

_TEXT = ["The Central Bank sets the Policy Rate, which affects GDP."]


def _build(client: TestClient) -> dict[str, Any]:
    resp = client.post("/build-kb", json={"texts": _TEXT})
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def test_a_built_ontology_starts_as_a_draft_with_a_review_list() -> None:
    built = _build(TestClient(create_app()))
    assert built["state"] == "draft"
    assert built["ontology_id"]
    assert built["review"], "a generated ontology should have review items, not zero"


def test_a_draft_cannot_authorize_an_answer() -> None:
    client = TestClient(create_app())
    built = _build(client)
    resp = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": built["kb_id"]},
    )
    assert resp.status_code == 409
    assert "not 'published'" in resp.json()["detail"]


def test_publishing_requires_passing_c_omega_first() -> None:
    client = TestClient(create_app())
    built = _build(client)
    oid = built["ontology_id"]
    # straight to publish, skipping review -> refused: the record is still a draft
    resp = client.post(f"/ontology/{oid}/publish", json={"version": "v1"})
    assert resp.status_code == 409
    assert "must pass CΩ" in resp.json()["detail"]


def test_an_edit_that_c_omega_rejects_names_the_rule() -> None:
    client = TestClient(create_app())
    oid = _build(client)["ontology_id"]
    bad = "version: v1\nentities:\n  - type: A\n    subtype_of: DoesNotExist\n"
    resp = client.put(f"/ontology/{oid}", json={"ontology_yaml": bad})
    assert resp.status_code == 400
    assert "DoesNotExist" in resp.json()["detail"]  # the reviewer is told what to fix


def test_the_full_path_draft_to_validated_to_published() -> None:
    client = TestClient(create_app())
    built = _build(client)
    oid = built["ontology_id"]

    edited = client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    assert edited.status_code == 200
    assert edited.json()["state"] == "validated"
    assert edited.json()["can_authorize_answers"] is False  # validated is still not approved

    published = client.post(f"/ontology/{oid}/publish", json={"version": "macro-v1"})
    assert published.status_code == 200
    body = published.json()
    assert body["state"] == "published"
    assert body["version"] == "macro-v1"
    assert body["can_authorize_answers"] is True
    assert "version: macro-v1" in body["ontology_yaml"]  # the cited version is in the ontology

    # and now the same question is answerable
    ans = client.post(
        "/answer",
        json={"query_id": "q1", "question": "Give the GDP reading.", "kb_id": built["kb_id"]},
    )
    assert ans.status_code == 200, ans.text


def test_a_published_ontology_is_frozen() -> None:
    client = TestClient(create_app())
    built = _build(client)
    oid = built["ontology_id"]
    client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    client.post(f"/ontology/{oid}/publish", json={"version": "macro-v1"})

    resp = client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    assert resp.status_code == 409
    assert "frozen" in resp.json()["detail"]  # edit it by publishing a new version


def test_history_records_every_transition() -> None:
    client = TestClient(create_app())
    built = _build(client)
    oid = built["ontology_id"]
    client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    client.post(f"/ontology/{oid}/publish", json={"version": "macro-v1"})
    history = client.get(f"/ontology/{oid}").json()["history"]
    assert len(history) == 3
    assert "draft" in history[0] and "validated" in history[1] and "published" in history[2]


# ---- the review checklist names what a builder cannot know ----

def test_checklist_flags_a_quantity_typed_as_string() -> None:
    items = review_checklist(
        "version: v1\nentities:\n  - type: Country\n    properties:\n"
        "      - {name: gdp_per_capita, type: string}\n"
    )
    kinds = {i["kind"]: i for i in items}
    assert "suspect_base_type" in kinds
    assert "gdp_per_capita" in kinds["suspect_base_type"]["target"]


def test_checklist_flags_missing_guards_and_default_cardinality() -> None:
    yaml = (
        "version: v1\n"
        "entities:\n  - type: Country\n  - type: Region\n"
        "verbs:\n  - {name: ACT, class: institutional}\n"
        "relations:\n  - {name: sits_in, from: Country, to: Region}\n"
        "actions:\n  - {name: DoIt, verb: ACT, target: Country}\n"
    )
    kinds = {i["kind"] for i in review_checklist(yaml)}
    assert "default_cardinality" in kinds   # many_to_many was never confirmed
    assert "missing_guard" in kinds         # a precondition is a governance decision
    assert "missing_effect" in kinds
    assert "assign_method_roles" in kinds   # outcome/dial/control is a modelling choice


def test_checklist_reports_an_ontology_that_does_not_load() -> None:
    items = review_checklist("version: v1\nentities:\n  - type: A\n    subtype_of: Nope\n")
    assert items[0]["kind"] == "does_not_load"
    assert "Nope" in items[0]["detail"]
