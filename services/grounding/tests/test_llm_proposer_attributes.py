"""The model proposer names attributes, and names them as the question put them.

Without this the attribute check was reachable only from the keyword proposer. Turning on the
model proposer — which is what a deployment does — silently restored the behaviour the check
was added to remove: the question resolved to the entity and the answer was every value held
about it, including for an attribute the ontology does not declare.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from loka_grounding.llm_proposer import LLMProposer


class _Client:
    """Records what was sent and replies with whatever the test scripted."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.system = ""
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, system: str, **_: object) -> object:
        self.system = system
        text = json.dumps(self.reply)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_attributes_are_proposed() -> None:
    client = _Client({"task_type": "descriptive", "targets": ["Seller"],
                      "attributes": ["on_time_rate"]})
    proposal = LLMProposer(["Seller"], client=client).propose("the seller's on-time rate?")
    assert proposal.targets == ("Seller",)
    assert proposal.attributes == ("on_time_rate",)


def test_an_attribute_outside_the_ontology_is_reported_not_dropped() -> None:
    """The opposite instruction from the one that governs targets, and the reason is asymmetric:
    a hallucinated entity should never reach the binder, while an attribute the ontology lacks
    must reach it — that is the only way the question gets refused rather than answered with
    everything else about the entity."""
    client = _Client({"task_type": "descriptive", "targets": ["Seller"],
                      "attributes": ["credit score"]})
    proposal = LLMProposer(["Seller"], client=client).propose("the seller's credit score?")
    assert proposal.attributes == ("credit score",)


def test_the_instruction_asks_for_attributes_in_the_question_s_own_wording() -> None:
    client = _Client({"task_type": "descriptive", "targets": [], "attributes": []})
    LLMProposer(["Seller"], client=client).propose("anything")
    assert "attributes" in client.system
    assert "whether or not it appears in any list" in client.system


def test_a_reply_without_attributes_still_binds() -> None:
    """Older replies, and questions about an entity in general, carry none — which is not the
    same as naming one that does not exist."""
    client = _Client({"task_type": "descriptive", "targets": ["Seller"]})
    assert LLMProposer(["Seller"], client=client).propose("tell me about sellers").attributes == ()
