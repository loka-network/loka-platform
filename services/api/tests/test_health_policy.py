"""The decision half for health: projection -> scenarios -> decision memo + audit."""

from __future__ import annotations

from loka_api.health_policy import evaluate_and_decide, evaluate_scenarios, select_and_decide

_PROJ = {
    "outcome": "under5_mortality", "dial": "health_exp_per_capita",
    "current_dial": 90.0, "new_dial": 150.0,
    "current_outcome": 49.1, "projected_outcome": 48.551,
    "interval_95": [40.2, 56.9], "identification": "observational",
    "effect": -0.549, "effect_interval_95": [-0.9, -0.2],  # excludes 0 -> significant
    "fit": {"n": 4382, "params": 8, "r2": 0.809, "sample_digest": "11cfefa614c36ddc"},
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


def test_a_determined_direction_over_a_weak_fit_reads_differently() -> None:
    """Two failures must not read alike: "cannot tell" and "can tell, and it explains little"."""
    weak = {**_PROJ, "fit": {"n": 96454, "params": 4, "r2": 0.013, "sample_digest": "d"}}
    d = evaluate_and_decide(weak, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]
    assert d["effect_distinguishable_from_zero"] is True     # the direction is determined
    assert d["explanatory_power"]["explains_little"] is True  # but it accounts for little
    assert "explains little of any individual case" in d["recommendation"]


def test_a_strong_fit_carries_no_weak_fit_qualifier() -> None:
    d = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]
    assert d["explanatory_power"]["r2"] == 0.809
    assert d["explanatory_power"]["explains_little"] is False
    assert "explains little" not in d["recommendation"]


def test_the_weak_fit_threshold_is_published_not_applied_silently() -> None:
    d = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]
    assert d["explanatory_power"]["weak_fit_threshold"] == 0.10  # a reader can disagree with it


def test_audit_hash_is_deterministic_for_replay() -> None:
    a = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    b = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    assert a == b  # same inputs -> same hash (replayable)


def test_audit_hash_binds_the_fitted_sample() -> None:
    """A data revision must change the hash — otherwise two different answers share one audit id."""
    a = evaluate_and_decide({**_PROJ, "fit": {"n": 4382, "params": 8, "sample_digest": "aaa"}},
                          iso="ZMB", ontology_version="health-v1", guard="g", method_name="m")
    b = evaluate_and_decide({**_PROJ, "fit": {"n": 4382, "params": 8, "sample_digest": "bbb"}},
                          iso="ZMB", ontology_version="health-v1", guard="g", method_name="m")
    assert a["decision"]["audit_manifest"] != b["decision"]["audit_manifest"]


def test_audit_inputs_are_published_so_the_hash_can_be_recomputed() -> None:
    import hashlib

    from loka_api.health_policy import _audit_preimage

    d = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]
    recomputed = hashlib.sha256(_audit_preimage(d["audit_inputs"]).encode()).hexdigest()[:16]
    assert recomputed == d["audit_manifest"]  # an auditor can verify it independently
    assert d["audit_inputs"]["sample_digest"] is not None
    assert d["audit_inputs"]["replayable"] is True


def test_a_decision_without_a_sample_digest_is_marked_not_replayable() -> None:
    proj = {**_PROJ, "fit": {"n": 10, "params": 3}}  # no digest
    d = evaluate_and_decide(proj, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]
    assert d["audit_inputs"]["replayable"] is False  # visible, not silently degraded


def test_audit_hash_changes_with_ontology_version() -> None:
    a = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    b = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v2",
                          guard="g", method_name="m")["decision"]["audit_manifest"]
    assert a != b  # a different Ω -> a different audit trail


def test_observational_evidence_may_not_justify_a_recommendation() -> None:
    """The identification label is applied, not merely reported: a projection fitted on observed
    data is correlational, and the admissibility matrix does not admit that as grounds to act."""
    d = evaluate_and_decide(_PROJ, iso="ZMB", ontology_version="health-v1",
                            guard="g", method_name="m")["decision"]
    a = d["admissibility"]
    assert a["identification"] == "observational"
    assert a["may_justify_a_recommendation"] is False
    assert a["admissible_for"] == ["forecast_conditioning"]
    assert "does not accept as the justification for a recommendation" in d["recommendation"]


def test_quasi_experimental_evidence_carries_no_such_caveat() -> None:
    """A design that does identify the effect is admitted, and the memo says nothing extra."""
    identified = {**_PROJ, "identification": "quasi_experimental"}
    d = evaluate_and_decide(identified, iso="ZMB", ontology_version="health-v1",
                            guard="g", method_name="m")["decision"]
    assert d["admissibility"]["may_justify_a_recommendation"] is True
    assert "block_a_justification" in d["admissibility"]["admissible_for"]
    assert "admissibility matrix" not in d["recommendation"]


def test_a_simulator_derived_claim_is_admitted_for_nothing() -> None:
    """Output of the simulator is quarantined: it conditions nothing and justifies nothing."""
    simulated = {**_PROJ, "identification": "simulator_derived"}
    a = evaluate_and_decide(simulated, iso="ZMB", ontology_version="health-v1",
                            guard="g", method_name="m")["decision"]["admissibility"]
    assert a["admissible_for"] == []
    assert a["may_justify_a_recommendation"] is False


def test_an_unrecognised_label_is_reported_rather_than_assumed_safe() -> None:
    a = evaluate_and_decide({**_PROJ, "identification": "vibes"}, iso="ZMB",
                            ontology_version="health-v1", guard="g",
                            method_name="m")["decision"]["admissibility"]
    assert a["recognised"] is False
    assert a["may_justify_a_recommendation"] is False
