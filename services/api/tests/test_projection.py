"""Controlled projection: recovers a known relationship, anchors to the target, holds controls."""

from __future__ import annotations

import math

from loka_api.projection import controlled_projection


def _panel() -> list[dict[str, float]]:
    # Known truth: outcome = 100 - 20*ln(spend) + 0.5*ctrl  (no noise) so the fit is exact.
    rows = []
    for spend in (10, 20, 40, 80, 160, 320, 640):
        for ctrl in (0.0, 10.0, 20.0):
            rows.append(
                {"spend": spend, "ctrl": ctrl, "y": 100 - 20 * math.log(spend) + 0.5 * ctrl}
            )
    return rows


def test_recovers_relationship_and_projects() -> None:
    panel = _panel()
    target = {"spend": 80, "ctrl": 10.0, "y": 100 - 20 * math.log(80) + 0.5 * 10.0}
    res = controlled_projection(
        panel, outcome="y", dial="spend", controls=["ctrl"], target=target,
        new_dial=160, log_cols=["spend"],
    )
    # perfect fit
    assert res["fit"]["r2"] > 0.999
    # truth at spend=160: 100 - 20*ln(160) + 5
    expected = 100 - 20 * math.log(160) + 0.5 * 10.0
    assert abs(res["projected_outcome"] - expected) < 0.1
    assert res["identification"] == "observational"


def test_anchors_to_target_current_value() -> None:
    panel = _panel()
    # target with an offset from the model (real value 5 below the fitted line)
    fitted = 100 - 20 * math.log(80) + 0.5 * 10.0
    target = {"spend": 80, "ctrl": 10.0, "y": fitted - 5.0}
    # projecting at the SAME dial must return the target's real value (anchoring)
    res = controlled_projection(
        panel, outcome="y", dial="spend", controls=["ctrl"], target=target,
        new_dial=80, log_cols=["spend"],
    )
    assert abs(res["projected_outcome"] - (fitted - 5.0)) < 1e-2  # exact up to 3-dp rounding
    assert res["controls_held_fixed"] == {"ctrl": 10.0}


def test_clamp_min() -> None:
    panel = _panel()
    target = {"spend": 80, "ctrl": 10.0, "y": 100 - 20 * math.log(80) + 5.0}
    res = controlled_projection(
        panel, outcome="y", dial="spend", controls=["ctrl"], target=target,
        new_dial=10**9, log_cols=["spend"], clamp_min=0.0,
    )
    assert res["projected_outcome"] >= 0.0
    assert res["interval_95"][0] >= 0.0
