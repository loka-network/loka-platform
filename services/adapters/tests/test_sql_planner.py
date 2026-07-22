"""Tests for the constrained SQL planner (the safe text-to-SQL core)."""

from __future__ import annotations

import pytest
from loka_adapters import SqlPlanError, plan_select


def test_basic_select() -> None:
    sql, params = plan_select("iati_grants", ["id", "amount_usd", "status"])
    assert sql == "SELECT id, amount_usd, status FROM iati_grants"
    assert params == []


def test_filters_are_parameterised() -> None:
    sql, params = plan_select(
        "iati_grants", ["id", "amount_usd"], filters={"country": "ZMB"}, limit=10
    )
    assert sql == "SELECT id, amount_usd FROM iati_grants WHERE country = %s LIMIT 10"
    assert params == ["ZMB"]  # value bound, never interpolated


def test_injection_attempt_rejected_in_table() -> None:
    with pytest.raises(SqlPlanError):
        plan_select("grants; DROP TABLE users", ["id"])


def test_injection_attempt_rejected_in_column() -> None:
    with pytest.raises(SqlPlanError):
        plan_select("grants", ["id", "amount) --"])


def test_empty_columns_rejected() -> None:
    with pytest.raises(SqlPlanError):
        plan_select("grants", [])


def test_negative_limit_rejected() -> None:
    with pytest.raises(SqlPlanError):
        plan_select("grants", ["id"], limit=-1)
