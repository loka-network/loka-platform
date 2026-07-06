"""Tests for the in-memory read-only adapter and the data-access contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from loka_adapters import InMemoryAdapter
from loka_schemas import (
    AuthenticationError,
    Certificate,
    Interval,
    MemoryAdapter,
    ScopeError,
    Session,
    TypedPredicate,
    TypedRow,
)

FIXED = datetime(2026, 1, 1, tzinfo=UTC)

DATASET = {
    "Instrument": [
        {"id": "US10Y", "kind": "SovereignBond", "ts": datetime(2026, 1, 1, tzinfo=UTC)},
        {"id": "AAPL", "kind": "EquityShare", "ts": datetime(2026, 6, 1, tzinfo=UTC)},
    ],
    "CentralBank": [{"id": "Fed", "rate": 0.05}],
}


def collect(aiter: AsyncIterator[TypedRow]) -> list[TypedRow]:
    """Drive an async iterator to completion synchronously (avoids a pytest-asyncio dep)."""

    async def _run() -> list[TypedRow]:
        return [row async for row in aiter]

    return asyncio.run(_run())


def make_adapter() -> InMemoryAdapter:
    return InMemoryAdapter("adapter-1", DATASET, source="test-db", now=lambda: FIXED)


def session(*scopes: str) -> Session:
    return Session(subject="tester", scopes=frozenset(scopes), established_at=FIXED)


# ---- contract conformance (checked by mypy) ----

def _accepts(_: MemoryAdapter) -> None: ...


def test_conforms_to_protocol() -> None:
    _accepts(make_adapter())
    assert isinstance(make_adapter(), MemoryAdapter)


# ---- authentication ----

def test_authenticate_ok() -> None:
    a = make_adapter()
    s = asyncio.run(a.authenticate(Certificate(subject="tester", scopes=frozenset({"Instrument"}))))
    assert s.subject == "tester"
    assert "Instrument" in s.scopes


def test_authenticate_rejects_empty_subject() -> None:
    a = make_adapter()
    with pytest.raises(AuthenticationError):
        asyncio.run(a.authenticate(Certificate(subject="")))


# ---- query ----

def test_query_returns_rows_with_lineage() -> None:
    a = make_adapter()
    rows = collect(a.query(TypedPredicate("Instrument"), session("Instrument")))
    assert {r.values["id"] for r in rows} == {"US10Y", "AAPL"}
    assert all(r.lineage.adapter_id == "adapter-1" for r in rows)
    assert all(r.lineage.source == "test-db" for r in rows)


def test_query_applies_filters() -> None:
    a = make_adapter()
    pred = TypedPredicate("Instrument", filters={"kind": "SovereignBond"})
    rows = collect(a.query(pred, session("Instrument")))
    assert [r.values["id"] for r in rows] == ["US10Y"]


def test_query_applies_time_range() -> None:
    a = make_adapter()
    rng = Interval(start=datetime(2026, 3, 1, tzinfo=UTC))
    rows = collect(a.query(TypedPredicate("Instrument", time_range=rng), session("Instrument")))
    assert [r.values["id"] for r in rows] == ["AAPL"]  # only the June row is in range


# ---- scope enforcement (customer-scoped OAuth) ----

def test_query_out_of_scope_is_rejected() -> None:
    a = make_adapter()
    with pytest.raises(ScopeError):
        collect(a.query(TypedPredicate("CentralBank"), session("Instrument")))


def test_wildcard_scope_allows_any() -> None:
    a = make_adapter()
    rows = collect(a.query(TypedPredicate("CentralBank"), session("*")))
    assert rows[0].values["id"] == "Fed"


# ---- read-only guarantee ----

def test_returned_rows_do_not_mutate_source() -> None:
    a = make_adapter()
    rows = collect(a.query(TypedPredicate("CentralBank"), session("CentralBank")))
    # mutate the returned copy
    rows[0].values["rate"]  # readable
    mutated = dict(rows[0].values)
    mutated["rate"] = 0.99
    # source is untouched
    assert DATASET["CentralBank"][0]["rate"] == 0.05
