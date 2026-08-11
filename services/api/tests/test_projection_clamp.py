"""Whether an outcome can go negative is a property of the domain, not of the method."""

from __future__ import annotations

from loka_api.projection import controlled_projection

# A panel whose outcome is legitimately signed: delivery lateness, negative when early.
_PANEL = [
    {"days_late": str(-6.0 + i * 0.4), "weight_g": str(500 + i * 90), "freight": str(8 + i * 0.5)}
    for i in range(40)
]


def test_a_signed_outcome_is_not_floored_by_default() -> None:
    target = _PANEL[0]
    out = controlled_projection(
        _PANEL, outcome="days_late", dial="weight_g", controls=["freight"],
        target=target, new_dial=4000.0, log_cols=["weight_g"],
    )
    assert out["current_outcome"] < 0
    assert out["projected_outcome"] < 0          # an early delivery stays early
    assert out["interval_95"][0] < 0


def test_a_caller_that_knows_the_domain_can_floor_it() -> None:
    out = controlled_projection(
        _PANEL, outcome="days_late", dial="weight_g", controls=["freight"],
        target=_PANEL[0], new_dial=4000.0, log_cols=["weight_g"], clamp_min=0.0,
    )
    assert out["projected_outcome"] == 0.0       # the bound is the caller's to impose
