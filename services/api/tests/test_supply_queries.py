"""The supply ontology, asked questions in natural language.

The query path was never domain-specific — it reads the entity types off whatever engine the
world carries — but this ontology lived only behind its own endpoints, so nothing could reach
it this way and the query chapter of the paper had to draw its examples from another dataset.

Two failures these cover, both of which were live: a question naming an attribute the ontology
does not declare was answered rather than refused, and a question naming one it does declare
returned every attribute of every instance.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def _ask(question: str) -> dict:
    client = TestClient(create_app())
    resp = client.post(
        "/answer", json={"query_id": "q1", "question": question, "kb_id": "supply"}
    )
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


def test_the_supply_ontology_can_be_asked_a_question() -> None:
    body = _ask("What is the on_time_rate of the Seller?")
    assert body["formalized_query"]["targets"] == ["Seller"]
    assert body["retrieval"]["act"] == "asks"
    assert body["retrieval"]["facts"], "the seller rows should be in state"


def test_an_attribute_the_ontology_does_not_declare_is_refused() -> None:
    """Not answered with everything else about the entity. A question about a property that does
    not exist is a different failure from one about an entity that does not, and the caller acts
    on the difference — so it carries its own code rather than a 400 with a message to parse."""
    body = _ask("What is the credit_score of the Seller?")
    assert body["answer"] == "don't know"
    assert body["reason_code"] == "not_in_ontology"
    assert "credit_score" in body["reason"]
    assert "supply-v2" in body["reason"]


def test_the_answer_is_narrowed_to_what_was_asked_for() -> None:
    """1,121 sellers with four attributes each is 4,484 values, and returning all of them in
    reply to a question about one of them answers a question nobody asked."""
    body = _ask("What is the on_time_rate of the Seller?")
    facts = body["retrieval"]["facts"]
    assert facts
    assert all(k.endswith(".on_time_rate") for k in facts), sorted(facts)[:3]


def test_a_question_naming_no_attribute_still_returns_the_slice() -> None:
    """Narrowing applies to what was asked for; asking about the entity itself is not a request
    for nothing."""
    facts = _ask("Tell me about the Seller.")["retrieval"]["facts"]
    assert len({k.rsplit(".", 1)[-1] for k in facts}) > 1
