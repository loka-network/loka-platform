"""Acceptance tests for grounding: propose (keyword) + bind (deterministic validation).

The LLM proposer is not exercised here (it needs credentials); the binder is proposer-agnostic,
so validating the keyword proposer's output covers the load-bearing, verifiable path.
"""

from __future__ import annotations

import pytest
from loka_grounding import (
    EmptyProposal,
    KeywordProposer,
    QueryProposal,
    UnknownTarget,
    UnknownTaskType,
    bind,
    ground,
)


class FakeOntology:
    """Minimal OntologyView: the aid-domain entity types."""

    version = "aid-v0.1"
    _entities = frozenset(
        {"Funder", "Grant", "Program", "Region", "BeneficiaryGroup", "Outcome", "ExternalFactor"}
    )

    def has_entity(self, name: str) -> bool:
        return name in self._entities


SYNONYMS = {"funding": "Grant", "project": "Program", "country": "Region", "result": "Outcome"}


@pytest.fixture
def ontology() -> FakeOntology:
    return FakeOntology()


@pytest.fixture
def proposer(ontology: FakeOntology) -> KeywordProposer:
    return KeywordProposer(sorted(FakeOntology._entities), synonyms=SYNONYMS)


# ---- keyword proposer ----

def test_keyword_matches_entities_and_synonyms(proposer: KeywordProposer) -> None:
    p = proposer.propose("How does grant funding affect the outcome for this program?")
    assert "Grant" in p.targets
    assert "Program" in p.targets
    assert "Outcome" in p.targets


def test_task_type_counterfactual_beats_ranking(proposer: KeywordProposer) -> None:
    # has both "if" (counterfactual) and "which/more" (ranking) — counterfactual wins
    p = proposer.propose("If we fund program A instead of B, which raises the outcome more?")
    assert p.task_type == "counterfactual"


def test_task_type_defaults_to_descriptive(proposer: KeywordProposer) -> None:
    p = proposer.propose("Show the grant amounts by region.")
    assert p.task_type == "descriptive"


# ---- binder (deterministic validation) ----

def test_ground_end_to_end(proposer: KeywordProposer, ontology: FakeOntology) -> None:
    q = ground(
        "If we fund this program instead, which region benefits most?",
        proposer,
        ontology,
        query_id="g-1",
        signature="signed-by-g1",
    )
    assert q.query_id == "g-1"
    assert q.task_type == "counterfactual"
    assert set(q.targets) <= FakeOntology._entities
    assert q.signature == "signed-by-g1"


def test_unknown_target_rejected(ontology: FakeOntology) -> None:
    bad = QueryProposal(task_type="descriptive", targets=("Dragon",))
    with pytest.raises(UnknownTarget):
        bind(bad, ontology, query_id="g-2")


def test_unknown_task_type_rejected(ontology: FakeOntology) -> None:
    bad = QueryProposal(task_type="teleport", targets=("Grant",))
    with pytest.raises(UnknownTaskType):
        bind(bad, ontology, query_id="g-3")


def test_empty_proposal_rejected(ontology: FakeOntology) -> None:
    bad = QueryProposal(task_type="descriptive", targets=())
    with pytest.raises(EmptyProposal):
        bind(bad, ontology, query_id="g-4")


def test_question_naming_unknown_entity_grounds_to_known_only(
    proposer: KeywordProposer, ontology: FakeOntology
) -> None:
    # "weather" isn't in the ontology; the grant/outcome parts still ground cleanly
    q = ground(
        "How does weather and grant funding change the outcome?",
        proposer,
        ontology,
        query_id="g-5",
    )
    assert "Grant" in q.targets and "Outcome" in q.targets
    assert "weather" not in " ".join(q.targets).lower()
