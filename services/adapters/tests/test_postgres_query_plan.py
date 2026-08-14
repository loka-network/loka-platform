"""The Postgres adapter plans its reads through the validated planner, not by hand.

These exercise the statement the adapter would send, without needing a database: the point is
which path builds it, since a projection that Ω did not produce must not be expressible.
"""

from __future__ import annotations

import pytest
from loka_adapters.sql_planner import SqlPlanError
from loka_schemas.data import Interval, TypedPredicate


def _adapter() -> object:
    pytest.importorskip("psycopg")
    from loka_adapters.postgres import PostgresAdapter

    return PostgresAdapter("postgresql://unused", adapter_id="t", tables={"GDP": "gdp_state"})


def test_a_projection_from_the_ontology_is_planned_not_hand_written() -> None:
    a = _adapter()
    stmt, params = a._build_query(  # type: ignore[attr-defined]
        "gdp_state", TypedPredicate("GDP", columns=("value", "unit"))
    )
    assert stmt == "SELECT value, unit FROM gdp_state"   # a str: the planner built it
    assert params == []


def test_filters_and_a_time_window_are_bound_parameters() -> None:
    a = _adapter()
    stmt, params = a._build_query(  # type: ignore[attr-defined]
        "gdp_state",
        TypedPredicate(
            "GDP",
            filters={"iso3": "THA"},
            time_range=Interval(start="2020-01-01", end="2021-01-01"),
            columns=("value",),
        ),
    )
    assert stmt == "SELECT value FROM gdp_state WHERE iso3 = %s AND ts >= %s AND ts < %s"
    assert params == ["THA", "2020-01-01", "2021-01-01"]


def test_a_column_that_is_not_an_identifier_cannot_be_read() -> None:
    a = _adapter()
    with pytest.raises(SqlPlanError, match="not a valid SQL identifier"):
        a._build_query(  # type: ignore[attr-defined]
            "gdp_state", TypedPredicate("GDP", columns=("value) --",))
        )


def test_without_a_projection_it_falls_back_to_select_star() -> None:
    a = _adapter()
    stmt, _ = a._build_query("gdp_state", TypedPredicate("GDP"))  # type: ignore[attr-defined]
    assert "SELECT *" in str(stmt)      # the un-modelled path, and visibly the weaker one
