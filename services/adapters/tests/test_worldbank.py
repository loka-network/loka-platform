"""Tests for the World Bank adapter — row mapping via an injected fetch (no network)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from loka_adapters import WorldBankAdapter
from loka_schemas import Certificate, ScopeError, Session, TypedPredicate, TypedRow
from loka_schemas.adapter import AdapterError

FIXED = datetime(2026, 1, 1, tzinfo=UTC)

_PROJECTS_JSON: dict[str, object] = {
    "projects": {
        "P133184": {
            "project_name": "Lusaka Transmission and Distribution Rehabilitation",
            "totalamt": "105,000,000",
            "projectstatusdisplay": "Closed",
            "countryname": ["Republic of Zambia"],
        },
        "P179380": {
            "project_name": "National Energy Advancement",
            "totalamt": "100,000,000",
            "projectstatusdisplay": "Active",
            "countryname": ["Republic of Zambia"],
        },
    }
}

_WDI_JSON: list[object] = [
    {"page": 1},
    [
        {"date": "2020", "value": 44.6},
        {"date": "2021", "value": None},  # missing value → skipped
        {"date": "2022", "value": 47.8},
    ],
]


def _fake_fetch(url: str) -> object:
    if "search.worldbank.org" in url:
        return _PROJECTS_JSON
    return _WDI_JSON


def collect(aiter: AsyncIterator[TypedRow]) -> list[TypedRow]:
    async def _run() -> list[TypedRow]:
        return [row async for row in aiter]

    return asyncio.run(_run())


def make_adapter() -> WorldBankAdapter:
    return WorldBankAdapter("wb-test", fetch=_fake_fetch, now=lambda: FIXED)


def session(*scopes: str) -> Session:
    return Session(subject="tester", scopes=frozenset(scopes), established_at=FIXED)


def test_projects_mapping() -> None:
    a = make_adapter()
    rows = collect(a.query(TypedPredicate("Program", {"country": "ZM"}), session("Program")))
    assert {r.values["id"] for r in rows} == {"P133184", "P179380"}
    lusaka = next(r for r in rows if r.values["id"] == "P133184")
    assert lusaka.values["amount_usd"] == 105_000_000.0  # comma stripped, float
    assert lusaka.values["status"] == "Closed"
    assert lusaka.lineage.source == "worldbank_projects"


def test_wdi_mapping_skips_missing() -> None:
    a = make_adapter()
    pred = TypedPredicate("Outcome", {"country": "ZMB", "indicator": "EG.ELC.ACCS.ZS"})
    rows = collect(a.query(pred, session("Outcome")))
    assert [r.values["as_of"] for r in rows] == ["2020", "2022"]  # 2021 (null) skipped
    assert rows[-1].values["value"] == 47.8
    assert rows[0].values["id"] == "ZMB:EG.ELC.ACCS.ZS:2020"
    assert rows[0].values["country"] == "ZMB"   # a panel must know whose row it is


def test_scope_enforced() -> None:
    a = make_adapter()
    with pytest.raises(ScopeError):
        collect(a.query(TypedPredicate("Program"), session("Outcome")))  # not scoped for Program


def test_unknown_entity_rejected() -> None:
    a = make_adapter()
    with pytest.raises(AdapterError):
        collect(a.query(TypedPredicate("Dragon"), session("*")))


def test_outcome_requires_indicator() -> None:
    a = make_adapter()
    with pytest.raises(AdapterError):
        collect(a.query(TypedPredicate("Outcome", {"country": "ZMB"}), session("Outcome")))


def test_authenticate_rejects_empty_subject() -> None:
    a = make_adapter()
    from loka_schemas import AuthenticationError

    with pytest.raises(AuthenticationError):
        asyncio.run(a.authenticate(Certificate(subject="")))
