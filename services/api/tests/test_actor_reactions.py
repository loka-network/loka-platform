"""Who acts in a simulation is a property of Ω, and which engine answered is reported.

Two things must hold. The actors come from the ontology's typing constraints, so adding a
constraint adds an actor and no list in this codebase decides who exists. And a general
assistant standing in for the behavior model is labelled as such: it is agreeable by
construction, so it under-produces the refusals and delays a simulation exists to find, and a
result from it cannot be read as a behavioural forecast.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loka_api.simulation import actor_reactions
from loka_ontology import OntologyEngine, load_ontology_str

_WITH_ACTORS = """
version: t
entities:
  - {type: Seller}
  - {type: Platform}
  - {type: Product}
verbs:
  - {name: SHIP, class: factual}
  - {name: SUSPEND, class: institutional}
constraints:
  - {verb: SHIP,    agent_must_be: Seller,   target_must_be: [Product]}
  - {verb: SUSPEND, agent_must_be: Platform, target_must_be: [Seller]}
"""

_NO_CONSTRAINTS = "version: t\nentities:\n  - {type: Seller}\n"


def _engine(yaml: str) -> OntologyEngine:
    return OntologyEngine(load_ontology_str(yaml))


def _stub_behavior(kind: str) -> object:
    engine = SimpleNamespace(act=lambda **_: "waits and re-quotes")
    return lambda persona: (engine, kind)


def test_the_actors_are_the_agents_omega_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loka_serving.persona_engine_for", _stub_behavior("behavior-model"))
    out = actor_reactions(_engine(_WITH_ACTORS), "a delivery slips by a week")
    assert [a["actor"] for a in out["actors"]] == ["Platform", "Seller"]
    # Product is an entity but never an agent, so it is not an actor
    assert "Product" not in [a["actor"] for a in out["actors"]]


def test_an_ontology_that_names_no_agent_has_no_actors() -> None:
    out = actor_reactions(_engine(_NO_CONSTRAINTS), "anything")
    assert out["actors"] == []
    assert out["engine"] == "none"
    assert "names no agent" in out["note"]


def test_a_stand_in_engine_is_reported_as_uncalibrated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loka_serving.persona_engine_for", _stub_behavior("general-llm"))
    out = actor_reactions(_engine(_WITH_ACTORS), "a delivery slips")
    assert out["engine"] == "general-llm"
    assert out["calibrated"] is False
    assert "under-states adversarial responses" in out["note"]


def test_the_trained_behavior_model_is_the_only_one_called_calibrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("loka_serving.persona_engine_for", _stub_behavior("behavior-model"))
    out = actor_reactions(_engine(_WITH_ACTORS), "a delivery slips")
    assert out["calibrated"] is True
    assert out["note"] == ""


def test_an_actor_that_fails_does_not_take_down_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(persona: object) -> object:
        raise RuntimeError("model unreachable")

    monkeypatch.setattr("loka_serving.persona_engine_for", _boom)
    out = actor_reactions(_engine(_WITH_ACTORS), "a delivery slips")
    assert all("error" in a for a in out["actors"])   # reported per actor, not raised
