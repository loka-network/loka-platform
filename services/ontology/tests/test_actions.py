"""Ω's 4th primitive: action types load, validate their verb/target, and are queryable."""

from __future__ import annotations

import pytest
from loka_ontology import OntologyEngine, OntologyLoadError, load_ontology_str

_ONTOLOGY = """
version: act-v1
entities:
  - {type: CentralBank}
  # policy_rate is declared because CutRate's guard reads it: a guard is checked against the
  # attributes its target actually has, so a target with none makes the act unreachable.
  - type: PolicyRate
    properties:
      - {name: policy_rate, type: double}
verbs:
  - {name: RATE_CHANGE, class: institutional}
actions:
  - name: CutRate
    verb: RATE_CHANGE
    target: PolicyRate
    guard: "policy_rate > 0"
    effect: "policy_rate decreases by 25bp"
"""


def test_action_type_loads_and_is_queryable() -> None:
    engine = OntologyEngine(load_ontology_str(_ONTOLOGY))
    actions = engine.action_types()
    assert len(actions) == 1
    a = actions[0]
    assert a.name == "CutRate"
    assert a.verb == "RATE_CHANGE"
    assert a.target == "PolicyRate"
    assert a.guard and a.effect


def test_action_with_undefined_verb_is_rejected() -> None:
    bad = _ONTOLOGY.replace("verb: RATE_CHANGE", "verb: NOPE")
    with pytest.raises(OntologyLoadError, match="undefined verb"):
        load_ontology_str(bad)


def test_action_with_undefined_target_is_rejected() -> None:
    bad = _ONTOLOGY.replace("target: PolicyRate", "target: Ghost")
    with pytest.raises(OntologyLoadError, match="undefined target"):
        load_ontology_str(bad)
