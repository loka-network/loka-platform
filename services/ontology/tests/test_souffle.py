"""Tests for the Soufflé-backed CΩ type checker.

Skipped when the ``souffle`` binary is not installed, so CI without Soufflé still passes.
Where Soufflé is present, its verdicts must match the pure-Python engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loka_ontology import (
    OntologyEngine,
    SouffleTypeChecker,
    load_ontology,
    souffle_available,
)

pytestmark = pytest.mark.skipif(
    not souffle_available(), reason="souffle binary not installed"
)

TOY = Path(__file__).parent.parent / "examples" / "toy_ontology.yaml"

CASES = [
    ("REGULATE", "CentralBank", "SovereignBond"),  # ok: CentralBank ⪯ Regulator
    ("REGULATE", "Regulator", "Instrument"),  # ok
    ("REGULATE", "Bond", "Instrument"),  # violation: Bond not a Regulator
    ("REGULATE", "CentralBank", "Regulator"),  # violation: target not Instrument/PolicyLever
    ("TRADE", "Bond", "Instrument"),  # ok: no constraint on TRADE
    ("FLY", "Regulator", "Instrument"),  # unknown verb
    ("REGULATE", "Nope", "Instrument"),  # unknown type
]


@pytest.fixture
def checker() -> SouffleTypeChecker:
    return SouffleTypeChecker(load_ontology(TOY))


def test_admissible_binding(checker: SouffleTypeChecker) -> None:
    assert checker.check_binding("REGULATE", "CentralBank", "SovereignBond").ok is True


def test_agent_violation(checker: SouffleTypeChecker) -> None:
    res = checker.check_binding("REGULATE", "Bond", "Instrument")
    assert res.ok is False
    assert res.reason is not None and "type_violation" in res.reason


def test_target_violation(checker: SouffleTypeChecker) -> None:
    res = checker.check_binding("REGULATE", "CentralBank", "Regulator")
    assert res.ok is False


def test_unconstrained_verb_allowed(checker: SouffleTypeChecker) -> None:
    assert checker.check_binding("TRADE", "Bond", "Instrument").ok is True


def test_unknown_verb(checker: SouffleTypeChecker) -> None:
    res = checker.check_binding("FLY", "Regulator", "Instrument")
    assert res.ok is False and res.reason is not None and "undefined verb" in res.reason


def test_unknown_type(checker: SouffleTypeChecker) -> None:
    res = checker.check_binding("REGULATE", "Nope", "Instrument")
    assert res.ok is False and res.reason is not None and "undefined type" in res.reason


def test_souffle_agrees_with_pure_python(checker: SouffleTypeChecker) -> None:
    engine = OntologyEngine(load_ontology(TOY))
    for verb, agent, target in CASES:
        souffle = checker.check_binding(verb, agent, target)
        python = engine.check_binding(verb, agent, target)
        assert souffle.ok == python.ok, f"disagree on {(verb, agent, target)}"


def test_batch_check(checker: SouffleTypeChecker) -> None:
    results = checker.check_bindings(CASES)
    assert results[("REGULATE", "CentralBank", "SovereignBond")].ok is True
    assert results[("REGULATE", "Bond", "Instrument")].ok is False
