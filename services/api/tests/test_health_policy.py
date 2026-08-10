"""Slide-6 right half for health: projection -> scenarios -> decision memo + audit."""

from __future__ import annotations

from loka_api.health_policy import evaluate_scenarios, select_and_decide, slide6_right_half

_PROJ = {
    "outcome": "under5_mortality", "dial": "health_exp_per_capita",
    "current_dial": 90.0, "new_dial": 150.0,
    "current_outcome": 49.1, "projected_outcome": 48.551,
    "interval_95": [40.2, 56.9], "identification": "observational",
}


def test_scenarios_read_off_the_interval() -> None:
    scen = evaluate_scenarios(_PROJ)
    kinds = {s["kind"]: s["under5_mortality"] for s in scen}
    assert kinds["nominal"] == 48.551
    assert kinds["adverse"] == 56.9   # upper bound = worse (higher mortality)
    assert kinds["favorable"] == 40.2  # lower bound = better


def test_decision_reports_improvement_and_enforces_ontology_guard() -> None:
    scen = evaluate_scenarios(_PROJ)
    d = select_and_decide(
        _PROJ, scen, iso="ZMB", ontology_version="health-v1",
        guard="health_exp_per_capita > 0", method_name="project_under5_mortality",
    )
    assert "reduces" in d["recommendation"]  # 48.551 < 49.1 -> improvement
    assert d["welfare_objective"] == "minimize under5_mortality"
    assert d["constraints_enforced"] == ["health_exp_per_capita > 0"]  # sourced from Ω
    assert d["constraint_satisfied"] is True
    assert d["identification"] == "observational"  # honesty label carried through
    assert len(d["audit_manifest"]) == 16


def test_audit_hash_is_deterministic_for_replay() -> None:
    a = slide6_right_half(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    b = slide6_right_half(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    assert a == b  # same inputs -> same hash (replayable)


def test_audit_hash_changes_with_ontology_version() -> None:
    a = slide6_right_half(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    b = slide6_right_half(_PROJ, iso="ZMB", ontology_version="health-v2",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    assert a != b  # a different Ω -> a different audit trail
