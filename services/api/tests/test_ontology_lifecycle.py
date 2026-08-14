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


def test_checklist_flags_missing_guards_and_undeclared_link_details() -> None:
    yaml = (
        "version: v1\n"
        "entities:\n  - type: Country\n  - type: Region\n"
        "verbs:\n  - {name: ACT, class: institutional}\n"
        "relations:\n  - {name: sits_in, from: Country, to: Region}\n"
        "actions:\n  - {name: DoIt, verb: ACT, target: Country}\n"
    )
    kinds = {i["kind"] for i in review_checklist(yaml)}
    assert "undeclared_cardinality" in kinds   # the ontology never stated one
    assert "undeclared_link_field" in kinds    # no 'via': the relation cannot be traversed
    assert "missing_guard" in kinds            # a precondition is a governance decision
    assert "missing_effect" in kinds
    assert "assign_method_roles" in kinds      # outcome/dial/control is a modelling choice


def test_checklist_does_not_flag_a_declared_cardinality_or_a_subtype_name() -> None:
    """Two false positives worth keeping out: a reviewer who learns to ignore the list is worse
    off than one with no list."""
    yaml = (
        "version: v1\n"
        "entities:\n"
        "  - type: Product\n    properties:\n"
        "      - {name: seller_id, type: string, description: 'link to Seller'}\n"
        "  - type: BulkyProduct\n    subtype_of: Product\n"
        "  - type: Seller\n    properties:\n"
        "      - {name: seller_id, type: string, description: 'identity'}\n"
        "relations:\n"
        "  - {name: sold_by, from: Product, to: Seller, via: seller_id, "
        "cardinality: many_to_many}\n"
    )
    kinds = {i["kind"] for i in review_checklist(yaml)}
    assert "undeclared_cardinality" not in kinds  # many_to_many was stated, not defaulted
    assert "undeclared_link_field" not in kinds   # via is declared
    assert "possible_synonyms" not in kinds       # BulkyProduct ⪯ Product is not a duplicate


def test_checklist_reports_an_ontology_that_does_not_load() -> None:
    items = review_checklist("version: v1\nentities:\n  - type: A\n    subtype_of: Nope\n")
    assert items[0]["kind"] == "does_not_load"
    assert "Nope" in items[0]["detail"]


def test_publishing_rebinds_the_world_to_the_reviewed_ontology() -> None:
    """Review is not a record-keeping exercise: what a reviewer fixed must be what runs.

    The world built at /build-kb holds the engine compiled from the draft. If publishing left it
    there, the gate would check the record's state while every query still ran against the text
    the reviewer had corrected — the lifecycle would be theatre.
    """
    client = TestClient(create_app())
    built = _build(client)
    oid, kb_id = built["ontology_id"], built["kb_id"]

    # the builder could not infer that GDP has a numeric value; the reviewer declares it
    reviewed = built["ontology_yaml"].replace(
        "  - type: GDP\n",
        "  - type: GDP\n    properties:\n      - {name: value, type: double}\n",
    )
    assert client.put(f"/ontology/{oid}", json={"ontology_yaml": reviewed}).status_code == 200
    published = client.post(f"/ontology/{oid}/publish", json={"version": "macro-v1"})
    assert published.json()["worlds_rebound"] == 1

    # the declaration the reviewer added is now the one ingestion checks against
    body = client.post(f"/kb/{kb_id}/ingest", json={"data": [
        {"entity": "GDP", "instance": "US", "property": "value", "value": 2.1},
    ]}).json()
    assert body["data_ingested"] == 1
    assert body["data_rejected"] == []


_ROWS = [
    {"seller_id": "s1", "seller_state": "SP", "on_time_rate": 0.91, "joined": "2018-03-01"},
    {"seller_id": "s2", "seller_state": "RJ", "on_time_rate": 0.74, "joined": "2019-06-11"},
]


def test_an_ontology_can_begin_from_data_and_enters_the_same_lifecycle() -> None:
    """Most customers have tables, not a document describing their domain. What is inferred from
    values is a guess, so it is a draft like any other — no authority until a person publishes."""
    client = TestClient(create_app())
    body = client.post(
        "/build-kb-from-data",
        json={"entity_type": "Seller", "backing": "sellers_table", "rows": _ROWS},
    ).json()

    assert body["state"] == "draft"
    assert body["can_authorize_answers"] is False
    assert body["source"] == "data:sellers_table"
    assert "backing: sellers_table" in body["ontology_yaml"]   # where the rows came from
    assert "type: double" in body["ontology_yaml"]             # inferred from the values
    assert body["review"], "inference leaves decisions a machine reading data cannot settle"


def test_inference_guesses_and_the_checklist_is_where_that_is_caught() -> None:
    """`joined` holds dates and is inferred as text. The guess is not hidden: it is a draft, and
    review is the step that corrects it."""
    client = TestClient(create_app())
    body = client.post(
        "/build-kb-from-data", json={"entity_type": "Seller", "rows": _ROWS}
    ).json()
    assert "name: joined\n    type: string" in body["ontology_yaml"]   # the wrong guess, visible
    assert body["state"] == "draft"                                    # and not authoritative


def test_building_from_no_rows_is_refused() -> None:
    client = TestClient(create_app())
    resp = client.post("/build-kb-from-data", json={"entity_type": "Seller", "rows": []})
    assert resp.status_code == 400


# ---- the checklist has to be read to be useful ----

def test_a_boolean_is_not_asked_which_currency_it_is_denominated_in() -> None:
    """Units belong to quantities. The question was being put to every attribute of every type,
    so a real extraction produced forty-one items asking whether booleans and timestamps were
    per-capita or nominal. A reviewer who learns the list contains nonsense stops reading it,
    and the findings that were worth reading go with it."""
    items = review_checklist(
        "version: v1\nentities:\n  - type: Seller\n    properties:\n"
        "      - {name: is_business, type: boolean}\n"
        "      - {name: joined_at, type: timestamp}\n"
        "      - {name: state, type: string}\n"
        "      - {name: punctuality, type: double}\n"
    )
    flagged = {
        t
        for i in items
        if i["kind"] == "missing_units"
        for t in i.get("targets", [i["target"]])
    }
    assert flagged == {"Seller.punctuality"}


def test_country_is_not_reported_as_a_probable_number() -> None:
    """'country' contains 'count'. Substring matching turned a correct field into a finding."""
    items = review_checklist(
        "version: v1\nentities:\n  - type: Seller\n    properties:\n"
        "      - {name: country, type: string}\n      - {name: on_time_rate, type: string}\n"
    )
    suspect = {i["target"] for i in items if i["kind"] == "suspect_base_type"}
    assert suspect == {"Seller.on_time_rate"}  # the real one is still caught


def test_a_systematic_gap_is_one_finding_not_thirty_one() -> None:
    """Extraction from prose never yields cardinality, so every relation lacks it. That is one
    fact about the method, and reporting it per relation buried the specific findings under
    sixty-two identical lines."""
    relations = "\n".join(
        f"  - {{name: r{i}, from: A, to: B}}" for i in range(31)
    )
    items = review_checklist(
        f"version: v1\nentities:\n  - type: A\n  - type: B\nrelations:\n{relations}\n"
    )
    card = [i for i in items if i["kind"] == "undeclared_cardinality"]
    assert len(card) == 1
    assert card[0]["count"] == 31
    assert len(card[0]["targets"]) == 31  # folded, not truncated
    assert len(items) < 10  # the list stays readable


def test_a_handful_is_still_itemised() -> None:
    """Aggregation is for systematic gaps. Three relations missing a link field is specific
    enough to act on one at a time, and naming them is more useful than counting them."""
    relations = "\n".join(f"  - {{name: r{i}, from: A, to: B}}" for i in range(3))
    items = review_checklist(
        f"version: v1\nentities:\n  - type: A\n  - type: B\nrelations:\n{relations}\n"
    )
    via = [i for i in items if i["kind"] == "undeclared_link_field"]
    assert len(via) == 3
    assert {i["target"] for i in via} == {"r0", "r1", "r2"}
