"""Tests for inferring a draft ontology from data (Palantir-style dataset → object type)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from loka_ontology import (
    BaseType,
    guess_primary_key,
    infer_base_type,
    infer_ontology_from_rows,
    load_ontology_str,
    to_yaml,
)

ROWS = [
    {"id": "US10Y", "notional": 1000.0, "count": 3, "active": True, "asof": date(2026, 1, 1)},
    {"id": "AAPL", "notional": 2.5, "count": 7, "active": False, "asof": date(2026, 2, 1)},
]


def test_infer_base_type() -> None:
    assert infer_base_type(["a", "b"]) == BaseType.STRING
    assert infer_base_type([1, 2, 3]) == BaseType.INTEGER
    assert infer_base_type([1, 2.5]) == BaseType.DOUBLE
    assert infer_base_type([True, False]) == BaseType.BOOLEAN
    assert infer_base_type([datetime(2026, 1, 1, tzinfo=UTC)]) == BaseType.TIMESTAMP
    assert infer_base_type([date(2026, 1, 1)]) == BaseType.DATE
    assert infer_base_type([None, None]) == BaseType.STRING  # unknown → string


def test_bool_not_treated_as_integer() -> None:
    assert infer_base_type([True, False]) != BaseType.INTEGER


def test_guess_primary_key_prefers_id() -> None:
    assert guess_primary_key(ROWS) == "id"


def test_guess_primary_key_none_when_not_unique() -> None:
    rows = [{"x": 1}, {"x": 1}]
    assert guess_primary_key(rows) is None


def test_infer_ontology_types_and_pk() -> None:
    onto = infer_ontology_from_rows("Instrument", ROWS, backing="warehouse.instruments")
    et = onto.entities["Instrument"]
    types = {p.name: p.base_type for p in et.properties}
    assert types == {
        "id": BaseType.STRING,
        "notional": BaseType.DOUBLE,
        "count": BaseType.INTEGER,
        "active": BaseType.BOOLEAN,
        "asof": BaseType.DATE,
    }
    assert et.backing == "warehouse.instruments"
    # the guessed PK (id) is marked required
    assert next(p for p in et.properties if p.name == "id").required is True


def test_yaml_draft_round_trips_through_loader() -> None:
    onto = infer_ontology_from_rows("Instrument", ROWS, backing="warehouse.instruments")
    yaml_text = to_yaml(onto)
    reloaded = load_ontology_str(yaml_text)
    et = reloaded.entities["Instrument"]
    assert et.backing == "warehouse.instruments"
    assert {p.name for p in et.properties} == {"id", "notional", "count", "active", "asof"}
    assert next(p for p in et.properties if p.name == "notional").base_type == BaseType.DOUBLE
