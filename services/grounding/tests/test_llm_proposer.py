"""Tests for the LLM proposer using a fake client — verifies wiring + JSON parsing offline.

A live model call needs credentials and is not exercised here; the fake client returns a
canned reply so the parse path and the propose→bind→q* wiring are covered deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from loka_grounding import UnknownTarget, bind
from loka_grounding.llm_proposer import LLMProposer

ENTITIES = ["Funder", "Grant", "Program", "Region", "Outcome"]


@dataclass
class FakeBlock:
    type: str
    text: str


@dataclass
class FakeResp:
    content: list[FakeBlock]


class FakeClient:
    """Minimal stand-in for anthropic.Anthropic: records the call, returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_kwargs: dict[str, object] = {}

    @property
    def messages(self) -> FakeClient:
        return self

    def create(self, **kwargs: object) -> FakeResp:
        self.last_kwargs = kwargs
        return FakeResp(content=[FakeBlock(type="text", text=self._reply)])


class FakeOntology:
    version = "aid-v0.1"

    def has_entity(self, name: str) -> bool:
        return name in set(ENTITIES)


def test_parses_clean_json() -> None:
    client = FakeClient('{"task_type": "counterfactual", "targets": ["Program", "Outcome"], '
                        '"rationale": "compares programs"}')
    p = LLMProposer(ENTITIES, client=client).propose("If we fund A instead of B, which is best?")
    assert p.task_type == "counterfactual"
    assert p.targets == ("Program", "Outcome")
    assert client.last_kwargs["model"] == "claude-opus-4-8"


def test_parses_json_wrapped_in_prose() -> None:
    # tolerant extraction: model added chatter around the object
    client = FakeClient('Sure! Here you go:\n{"task_type": "ranking", "targets": ["Grant"], '
                        '"rationale": "x"}\nHope that helps.')
    p = LLMProposer(ENTITIES, client=client).propose("Rank the grants.")
    assert p.task_type == "ranking"
    assert p.targets == ("Grant",)


def test_proposal_still_validated_by_binder() -> None:
    # the model hallucinates an entity; the binder rejects it — model proposes, types dispose
    client = FakeClient('{"task_type": "descriptive", "targets": ["Dragon"], "rationale": "x"}')
    proposal = LLMProposer(ENTITIES, client=client).propose("anything")
    with pytest.raises(UnknownTarget):
        bind(proposal, FakeOntology(), query_id="q1")
