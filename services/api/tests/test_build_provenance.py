"""A draft records what produced it, in enough detail to reproduce or dispute it.

Two people extracting an ontology from one document reach different results, and the interesting
question is never whose is bigger — it is what differed. A record that says only "built by an
LLM" cannot answer that, so a disagreement about output becomes a disagreement with no evidence
on either side.

The prompt is the part that matters most, because it is not a setting but the method: it decides
which parts of Ω can come out of a text at all. The default asks for entities, attributes,
relations, verbs and the DATA/METHODS needs. It does not ask for cardinality, guards or norms —
so no domain text, however well written, will produce them through this route. That is a property
of the procedure, and the record has to carry it.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from loka_api.app import create_app

_TEXT = "The Central Bank sets the Policy Rate, which affects GDP."


def _build(client: TestClient, **extra: Any) -> dict[str, Any]:
    resp = client.post("/build-kb", json={"texts": [_TEXT], **extra})
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def test_a_draft_says_what_made_it() -> None:
    prov = _build(TestClient(create_app()))["provenance"]
    assert prov["method"]
    assert "model" in prov and "prompt" in prov


def test_the_input_is_identified_without_being_stored() -> None:
    """A domain document can be long and can be confidential. What a reader needs is to confirm
    they are holding the same one, which a digest gives without the platform keeping a copy."""
    prov = _build(TestClient(create_app()))["provenance"]
    assert len(prov["input_digest"]) == 16
    assert prov["input_chars"] == len(_TEXT)
    assert _TEXT not in str(prov)


def test_the_same_input_digests_the_same_and_a_changed_one_does_not() -> None:
    client = TestClient(create_app())
    first = _build(client)["provenance"]["input_digest"]
    again = _build(client)["provenance"]["input_digest"]
    changed = client.post("/build-kb", json={"texts": [_TEXT + " Also inflation."]})
    assert first == again
    assert changed.json()["provenance"]["input_digest"] != first


def test_without_a_model_the_record_says_so_rather_than_leaving_it_blank() -> None:
    """The rule-based builder ran. Recording ``model: None`` is not the same as recording
    nothing: one states that no model was involved, the other is silence a reader must guess at.
    """
    prov = _build(TestClient(create_app()))["provenance"]
    if prov["model"] is None:
        assert "no model" in prov["method"]
        assert prov["prompt"] is None


def test_a_caller_supplied_prompt_is_recorded_as_such() -> None:
    """The prompt is a parameter so a domain can supply its own. Which one ran has to be
    distinguishable afterwards, or the override quietly rewrites history."""
    body = _build(TestClient(create_app()), system_prompt="Extract only institutions.")
    prov = body["provenance"]
    if prov["model"] is not None:  # only the LLM route takes a prompt
        assert prov["prompt"] == "Extract only institutions."
        assert prov["prompt_source"] == "caller"


def test_provenance_survives_review_and_publication() -> None:
    """Publishing must not erase where the draft came from. The published ontology is what
    authorises answers, so it is exactly the one whose origin has to remain auditable."""
    client = TestClient(create_app())
    built = _build(client)
    oid = built["ontology_id"]
    client.put(f"/ontology/{oid}", json={"ontology_yaml": built["ontology_yaml"]})
    published = client.post(f"/ontology/{oid}/publish", json={"version": "v1"})
    assert published.status_code == 200, published.text
    assert published.json()["provenance"]["input_digest"] == built["provenance"]["input_digest"]


def test_a_data_built_draft_is_distinguishable_from_a_text_built_one() -> None:
    rows = [{"seller_id": "s1", "on_time_rate": 0.9}, {"seller_id": "s2", "on_time_rate": 0.7}]
    client = TestClient(create_app())
    body = client.post(
        "/build-kb-from-data", json={"entity_type": "Seller", "rows": rows}
    ).json()
    assert body["source"] == "data:rows"
    assert _build(client)["source"].startswith("builder:")


def test_a_failed_model_build_is_reported_not_replaced_with_segmentation(
    monkeypatch: Any,
) -> None:
    """The failure this rule exists for. A reasoning model spent its whole budget thinking and
    returned nothing, so /build-kb quietly ran the rule-based builder instead — which splits
    prose on capitalised words and produced twenty-four "entity types" including What, Two,
    First, Everything and UnderBrazilian. That went into the lifecycle as an ordinary draft.

    A caller who asked for a model and is handed sentence fragments cannot tell, from the
    ontology alone, that anything went wrong; they will read it as what our extraction does.
    Answering a different question than the one asked is worse than not answering.
    """
    monkeypatch.setenv("LOKA_LLM_BUILD", "1")

    import loka_serving

    def _explode(_purpose: str) -> Any:
        raise RuntimeError("model returned no content after 4 attempts")

    monkeypatch.setattr(loka_serving, "llm_for", _explode)

    resp = TestClient(create_app()).post("/build-kb", json={"texts": [_TEXT]})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "no ontology was produced" in detail["error"]
    assert "no content after 4 attempts" in detail["cause"]  # the real cause, not a summary
    assert "LOKA_LLM_BUILD" in detail["note"]  # and how to ask for the other builder on purpose


def test_the_rule_based_builder_is_still_available_when_asked_for(monkeypatch: Any) -> None:
    """It is not being removed — it makes the pipeline runnable with no model at all. What
    changed is that it is only used when it was requested."""
    monkeypatch.delenv("LOKA_LLM_BUILD", raising=False)
    body = _build(TestClient(create_app()))
    assert body["builder"] == "keyword"
    assert "not extraction" in body["provenance"]["method"]
