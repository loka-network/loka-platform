"""Each refusal code, produced on the path a caller actually uses.

Four codes were documented. On the general query path two of them were produced: an entity the
ontology does not have arrived as `unformalizable` because neither proposer will name one, and
an attribute with no stored value came back as an empty result, which is the shape of an answer.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app
from loka_api.world import build_supply_world
from loka_state import WorldState


def _client_with_empty_supply() -> TestClient:
    """The supply ontology with nothing in state — declared attributes, no values."""
    world = build_supply_world()
    world.state = WorldState()
    app = create_app()
    app.state.kb_worlds["empty"] = world
    return TestClient(app)


def test_not_in_ontology_names_what_is_declared() -> None:
    body = TestClient(create_app()).post("/answer", json={
        "query_id": "q", "question": "What is the credit_score of the Seller?", "kb_id": "supply",
    }).json()
    assert body["reason_code"] == "not_in_ontology"
    assert "credit_score" in body["reason"]
    # The refusal says what the ontology does have, so a caller who used the wrong name can see
    # the right one without reading the ontology.
    assert "on_time_rate" in body["reason"]


def test_no_data_is_a_refusal_not_an_empty_result() -> None:
    """`facts: {}` reads as an answer. A caller cannot tell it from a question whose answer is
    genuinely nothing, and the two call for different actions."""
    body = _client_with_empty_supply().post("/answer", json={
        "query_id": "q", "question": "What is the on_time_rate of the Seller?", "kb_id": "empty",
    }).json()
    assert body["reason_code"] == "no_data"
    assert "holds no value" in body["reason"]


def test_no_data_does_not_fire_when_there_is_data() -> None:
    body = TestClient(create_app()).post("/answer", json={
        "query_id": "q", "question": "What is the on_time_rate of the Seller?", "kb_id": "supply",
    }).json()
    assert "reason_code" not in body
    assert len(body["retrieval"]["facts"]) > 0


def test_unformalizable_when_nothing_can_be_grounded() -> None:
    body = TestClient(create_app()).post("/answer", json={
        "query_id": "q", "question": "Hello there", "kb_id": "supply",
    }).json()
    assert body["reason_code"] == "unformalizable"


def test_the_model_free_entry_reaches_the_same_ontologies() -> None:
    """Bound to the default world, /compile could only compile against that one — so the entry
    this system offers as a way round the model was unusable for any other domain."""
    resp = TestClient(create_app()).post("/compile", json={
        "query_id": "q", "task_type": "descriptive", "targets": ["Seller"], "kb_id": "supply",
    })
    assert resp.status_code == 200, resp.text
    assert "state_package" in resp.json()
