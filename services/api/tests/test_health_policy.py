"""Slide-6 right half for health: projection -> scenarios -> decision memo + audit."""

from __future__ import annotations

from loka_api.health_policy import evaluate_scenarios, select_and_decide, slide6_right_half

_PROJ = {
    "outcome": "under5_mortality", "dial": "health_exp_per_capita",
    "current_dial": 90.0, "new_dial": 150.0,
    "current_outcome": 49.1, "projected_outcome": 48.551,
    "interval_95": [40.2, 56.9], "identification": "observational",
    "effect": -0.549, "effect_interval_95": [-0.9, -0.2],  # excludes 0 -> significant
}

_PROJ_NOT_SIGNIFICANT = {
    **_PROJ,
    "effect_interval_95": [-1.329, 0.23],  # straddles 0 -> no claimable direction
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
    assert d["effect_distinguishable_from_zero"] is True
    assert "reduces under5_mortality" in d["recommendation"]  # a direction may be claimed
    assert d["welfare_objective"] == "minimize under5_mortality"
    assert d["constraints_enforced"] == ["health_exp_per_capita > 0"]  # sourced from Ω
    assert d["constraint_satisfied"] is True
    assert d["identification"] == "observational"  # honesty label carried through
    assert len(d["audit_manifest"]) == 16


def test_memo_refuses_to_claim_a_direction_when_the_effect_straddles_zero() -> None:
    """An effect CI that includes 0 must not be written up as "reduces mortality"."""
    scen = evaluate_scenarios(_PROJ_NOT_SIGNIFICANT)
    d = select_and_decide(
        _PROJ_NOT_SIGNIFICANT, scen, iso="ZMB", ontology_version="health-v1",
        guard="health_exp_per_capita > 0", method_name="project_under5_mortality",
    )
    assert d["effect_distinguishable_from_zero"] is False
    assert d["recommendation"].startswith("No effect distinguishable from zero")
    assert "does not support claiming" in d["recommendation"]


def test_scenarios_use_the_effect_interval_not_the_level_interval() -> None:
    """Bounds must come from interval_95 (effect-implied), never the wide level PI."""
    proj = {**_PROJ, "interval_95": [47.8, 49.3], "level_prediction_interval_95": [14.2, 82.9]}
    kinds = {s["kind"]: s["under5_mortality"] for s in evaluate_scenarios(proj)}
    assert kinds["adverse"] == 49.3    # from interval_95
    assert kinds["favorable"] == 47.8
    assert kinds["adverse"] != 82.9    # not the level prediction interval


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
