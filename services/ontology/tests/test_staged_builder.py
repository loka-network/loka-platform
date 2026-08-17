"""Staged extraction, and the check that every proposal came from the document.

Two claims are tested. That splitting the task keeps each answer small — the reason single-shot
extraction fails on a real document is that one reply has to carry the whole ontology. And that a
concept the text does not support is reported rather than accepted, since a model asked for an
ontology will supply a plausible one whether or not the document determines it.

The second claim is the one worth being careful about: a grounding check that passes everything
is worse than none, because it certifies. So there is a test that it actually rejects.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from loka_ontology import OntologyBuildError, StagedLLMBuilder, load_ontology_str
from loka_ontology.builder import build
from loka_ontology.staged_builder import _appears_in

_TEXT = """
Independent sellers list items on our marketplace. A shopper pays once, and what arrives may
come in several parcels, because the items in one basket often come from different sellers.
Each line of a purchase is dispatched on its own. We show the shopper a promised date at
checkout. A parcel arriving after that date is late. Above a certain weight an item cannot go
on the standard service and needs freight; the team calls those bulky items.
"""


class _Scripted:
    """Replies per stage, keyed by a fragment of the instruction, so the test states what each
    stage was asked rather than depending on call order."""

    def __init__(self, replies: dict[str, dict[str, Any]]) -> None:
        self._replies = replies
        self.calls: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, system: str, **kw: Any) -> Any:
        for marker, reply in self._replies.items():
            if marker in system:
                self.calls.append(marker)
                text = json.dumps(reply)
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
        raise AssertionError(f"no scripted reply for: {system[:60]}")


_GOOD = {
    "List the entity types": {"entities": [
        {"name": "Seller", "subtype_of": None, "evidence": "Independent sellers list items"},
        {"name": "Shopper", "subtype_of": None, "evidence": "A shopper pays once"},
        {"name": "Item", "subtype_of": None, "evidence": "sellers list items"},
        {"name": "BulkyItem", "subtype_of": "Item", "evidence": "the team calls those bulky items"},
        {"name": "Purchase", "subtype_of": None, "evidence": "Each line of a purchase"},
        {"name": "PurchaseLine", "subtype_of": None, "evidence": "Each line of a purchase"},
    ]},
    "give the attributes": {"attributes": {
        "Item": [{"name": "weight", "type": "double", "evidence": "Above a certain weight"}],
        "Seller": [],
    }},
    "extract the relations": {"relations": [
        {"from": "Seller", "verb": "lists", "to": "Item",
         "evidence": "Independent sellers list items on our marketplace."},
        {"from": "Purchase", "verb": "contains", "to": "PurchaseLine",
         "evidence": "Each line of a purchase is dispatched on its own."},
    ]},
    "Classify each relation verb": {"verbs": [["LISTS", "factual"], ["CONTAINS", "factual"]]},
}


def _build(replies: dict[str, dict[str, Any]]) -> tuple[Any, StagedLLMBuilder]:
    client = _Scripted(replies)
    builder = StagedLLMBuilder(client=client, model="test")
    return builder.propose([_TEXT]), builder


# ---- decomposition ----

def test_each_stage_is_a_separate_call() -> None:
    """The reason to decompose: one reply carrying an entire ontology is what gets truncated."""
    _, builder = _build(_GOOD)
    stages = [c["stage"] for c in builder.stage_calls]
    assert stages[0] == "entities"
    assert any(s.startswith("attributes") for s in stages)
    assert "relations" in stages and "verbs" in stages


def test_attributes_are_asked_in_batches() -> None:
    """Twenty entity types in one attribute call recreates the problem decomposition solves."""
    many = {"entities": [
        {"name": f"T{i}", "subtype_of": None, "evidence": "sellers"} for i in range(20)
    ]}
    replies = dict(_GOOD, **{"List the entity types": many})
    _, builder = _build(replies)
    attribute_calls = [c for c in builder.stage_calls if c["stage"].startswith("attributes")]
    assert len(attribute_calls) > 1


def test_the_result_loads_through_c_omega() -> None:
    draft, _ = _build(_GOOD)
    spec = build([_TEXT], _ScriptedBuilder(draft))
    onto = load_ontology_str(spec.ontology_yaml)
    assert onto.entities["BulkyItem"].subtype_of == "Item"
    assert {r.name for r in onto.relations} == {"lists", "contains"}


class _ScriptedBuilder:
    def __init__(self, draft: Any) -> None:
        self._draft = draft

    def propose(self, texts: Any) -> Any:
        return self._draft


# ---- grounding ----

def test_a_concept_the_document_does_not_mention_is_reported() -> None:
    """The failure this exists for. A model asked for a marketplace ontology will offer
    warehouses and couriers whether or not the document has any."""
    invented = {"entities": [
        {"name": "Seller", "subtype_of": None, "evidence": "Independent sellers list items"},
        {"name": "Warehouse", "subtype_of": None,
         "evidence": "goods are held in regional warehouses"},
    ]}
    _, builder = _build(dict(_GOOD, **{"List the entity types": invented}))
    assert "Warehouse" in builder.grounding.ungrounded
    assert "Seller" in builder.grounding.grounded


def test_an_ungrounded_concept_is_kept_not_dropped() -> None:
    """It may be a correct generalisation the text words differently. Deleting it would replace
    a visible disagreement with a silent one."""
    invented = {"entities": [
        {"name": "Seller", "subtype_of": None, "evidence": "Independent sellers"},
        {"name": "Warehouse", "subtype_of": None, "evidence": "regional warehouses"},
    ]}
    draft, _ = _build(dict(_GOOD, **{"List the entity types": invented}))
    assert "Warehouse" in {e.name for e in draft.entities}


def test_a_generalised_name_still_counts_as_grounded_when_the_evidence_is_real() -> None:
    """"PurchaseLine" appears nowhere; "Each line of a purchase" does. Requiring the *name* in
    the text would reject every abstraction, which is the whole job."""
    _, builder = _build(_GOOD)
    assert "PurchaseLine" in builder.grounding.grounded


def test_a_relation_whose_sentence_is_not_in_the_text_is_reported() -> None:
    """The HT-R-O point: the model is asked what the text says, not what would be sensible. A
    fabricated sentence is the signal that it answered the second question."""
    made_up = {"relations": [
        {"from": "Seller", "verb": "insures", "to": "Item",
         "evidence": "Sellers must insure every item above a certain value."},
    ]}
    _, builder = _build(dict(_GOOD, **{"extract the relations": made_up}))
    assert "Seller -insures-> Item" in builder.grounding.ungrounded


def test_a_relation_between_types_that_were_never_proposed_is_recorded() -> None:
    """Stage 3 disagreeing with stage 1 is information. Dropping it silently would hide that the
    stages saw different domains."""
    stray = {"relations": [
        {"from": "Courier", "verb": "delivers", "to": "Parcel", "evidence": "several parcels"},
    ]}
    draft, builder = _build(dict(_GOOD, **{"extract the relations": stray}))
    assert draft.relations == ()
    assert any("Courier" in k for k in builder.grounding.ungrounded)


def test_the_grounding_check_can_actually_fail() -> None:
    """A check that certifies everything is worse than no check."""
    assert _appears_in("sellers list items", _TEXT)
    assert _appears_in("PurchaseLine", "each purchase line is dispatched")  # CamelCase split
    assert _appears_in("parcel", "several parcels arrive")                  # plural tolerated
    assert not _appears_in("regional warehouses", _TEXT)
    assert not _appears_in("", _TEXT)


def test_the_grounding_rate_is_reported() -> None:
    _, builder = _build(_GOOD)
    report = builder.grounding.as_dict()
    assert report["checked"] > 0
    assert 0.0 < report["rate"] <= 1.0
    assert report["evidence"]["Seller"] == "Independent sellers list items"


# ---- failures ----

def test_a_stage_returning_no_types_is_an_error_not_an_empty_ontology() -> None:
    with pytest.raises(OntologyBuildError, match="proposed no entity types"):
        _build(dict(_GOOD, **{"List the entity types": {"entities": []}}))


def test_a_truncated_stage_names_the_stage() -> None:
    """Which stage was cut off decides what to do about it, so the error says which."""
    class _Cut:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kw: Any) -> Any:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"entities": [{"na')]
            )

    with pytest.raises(OntologyBuildError, match="stage 'entities' was cut off"):
        StagedLLMBuilder(client=_Cut(), model="test").propose([_TEXT])


def test_the_prompts_are_readable_for_the_record() -> None:
    """What ran has to be recordable, and there are four of them now."""
    builder = StagedLLMBuilder(client=_Scripted(_GOOD), model="test")
    prompt = builder.system_prompt
    assert prompt.count("[stage ") == 4
    assert "evidence" in prompt
