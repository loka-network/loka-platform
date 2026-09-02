"""The three-role contract, enforced rather than described.

roles.md says the judgement is made by code against a frozen protocol and the role explains and
chooses. These cover the part that is code: whether a role may act, whether a handoff carries
what its contract names, and the one derivation the document fixes exactly.

What is not covered, because it does not exist: the map from measurements to the four
dispositions. roles.md freezes that by hand and never derives it, so there is nothing here to
test and nothing here to invent.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from loka_api.app import create_app


def _act(**body: object) -> dict:
    return dict(TestClient(create_app()).post("/act", json=body).json())


def test_a_role_cannot_perform_an_act_no_signature_gives_it() -> None:
    """Not a rule the chemist breaks — a sentence the ontology cannot write. The distinction
    matters because a rule can be argued with and a missing signature cannot."""
    body = _act(role="Chemist", verb="judge_target_relevance")
    assert body["permitted"] is False
    assert body["code"] == "out_of_role"
    assert "Biologist" in body["reason"]


def test_the_role_the_signature_names_may_perform_it() -> None:
    body = _act(role="Biologist", verb="judge_target_relevance")
    assert body["permitted"] is True


def test_synthesizability_belongs_to_the_chemist_alone() -> None:
    """The one funnel segment that bears on usable yield and that no metric covers."""
    assert _act(role="AIExpert", verb="judge_synthesizability")["code"] == "out_of_role"
    assert _act(role="Chemist", verb="judge_synthesizability")["permitted"] is True


def test_a_ceiling_above_95_blocks_the_report_and_names_the_norm() -> None:
    body = _act(role="AIExpert", verb="publish_report",
                state={"ceiling": 97.0, "size_matched_n": 40, "pockets_measured": 5,
                       "mixes_evidence_classes": 0})
    assert body["permitted"] is False
    assert body["norm"] == "CeilingAbove95IsAWarning"
    assert body["why"]


def test_the_main_metric_may_not_be_reported_without_the_size_matched_subset() -> None:
    """A full-set mean has been shown on this data to pick the wrong winner."""
    body = _act(role="AIExpert", verb="publish_report",
                state={"ceiling": 70.0, "size_matched_n": 0, "pockets_measured": 5,
                       "mixes_evidence_classes": 0})
    assert body["norm"] == "MainMetricNeedsSizeMatchedSubset"


def test_strain_energy_may_not_be_reported_as_a_mean() -> None:
    assert _act(role="Chemist", verb="report_geometry",
                state={"statistic": 2})["norm"] == "StrainEnergyIsAMedian"


def test_an_act_permitted_is_still_not_an_act_performed() -> None:
    assert _act(role="Biologist", verb="judge_target_relevance")["requires_confirmation"] is True


def test_a_handoff_missing_a_contract_field_is_refused() -> None:
    """The field list is the required attributes of the role's output type in the ontology, not
    a list in this endpoint — so what a complete handoff is can be read where it is declared."""
    body = TestClient(create_app()).post("/output", json={
        "role": "Biologist",
        "payload": {"output_id": "o1", "target_id": "t1", "hypothesis_id": "h1",
                    "citations_resolved": 2, "leakage_conclusion": "clear",
                    "confidence_evidence_class": 2},
    }).json()
    assert body["accepted"] is False
    assert "missing required property: falsifier" in body["problems"]
    assert "falsifier" in body["required_by_contract"]


def test_the_target_number_shows_its_arithmetic() -> None:
    """X + (100 − X)/3 — a closed fraction rather than a number of points, because points mean
    something different at every baseline."""
    body = TestClient(create_app()).post("/target-number", json={"baseline": 66.0}).json()
    assert body["target"] == 77.3333
    assert body["formula"] == "X + (100 - X) / 3"


def test_a_falsifier_is_evaluated_against_the_measurements() -> None:
    post = TestClient(create_app()).post
    assert post("/falsify", json={"condition": "recovery >= 77.33",
                                  "measurements": {"recovery": 80.1}}).json()["falsified"] is True
    assert post("/falsify", json={"condition": "recovery >= 77.33",
                                  "measurements": {"recovery": 71.2}}).json()["falsified"] is False


def test_an_unreadable_falsifier_is_neither_satisfied_nor_refuted() -> None:
    """Reporting it as either would be this code deciding something it cannot see."""
    body = TestClient(create_app()).post("/falsify", json={
        "condition": "if the results look bad", "measurements": {"recovery": 80.1}}).json()
    assert body["evaluable"] is False
    assert "falsified" not in body


def test_a_falsifier_naming_a_measurement_nobody_supplied_is_unevaluable() -> None:
    body = TestClient(create_app()).post("/falsify", json={
        "condition": "recovery >= 77.33", "measurements": {}}).json()
    assert body["evaluable"] is False
    assert "recovery" in body["reason"]
