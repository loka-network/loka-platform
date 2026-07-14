"""Integration tests for the Postgres adapter.

Skipped unless ``LOKA_PG_DSN`` points at a reachable Postgres (see infra/docker-compose.yml)
and the ``psycopg`` driver is installed. Run these on the cloud/dev database.
"""

from __future__ import annotations

import asyncio
import os

import pytest

psycopg = pytest.importorskip("psycopg")

from loka_adapters.postgres import PostgresAdapter  # noqa: E402
from loka_schemas import Certificate, TypedPredicate  # noqa: E402

DSN = os.environ.get("LOKA_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="LOKA_PG_DSN not set")


@pytest.fixture
def table() -> object:
    assert DSN is not None
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS loka_it_instrument")
        conn.execute(
            "CREATE TABLE loka_it_instrument (id text, kind text)"
        )
        conn.execute(
            "INSERT INTO loka_it_instrument VALUES ('US10Y','SovereignBond'),('AAPL','EquityShare')"
        )
    yield None
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS loka_it_instrument")


def _collect(adapter: PostgresAdapter, pred: TypedPredicate) -> list[dict[str, object]]:
    async def run() -> list[dict[str, object]]:
        cert = Certificate(subject="it", scopes=frozenset({"Instrument"}))
        session = await adapter.authenticate(cert)
        return [dict(row.values) async for row in adapter.query(pred, session)]

    return asyncio.run(run())


def test_query_reads_rows(table: object) -> None:
    assert DSN is not None
    adapter = PostgresAdapter(DSN, adapter_id="pg-it", tables={"Instrument": "loka_it_instrument"})
    rows = _collect(adapter, TypedPredicate("Instrument"))
    assert {r["id"] for r in rows} == {"US10Y", "AAPL"}


def test_query_filters(table: object) -> None:
    assert DSN is not None
    adapter = PostgresAdapter(DSN, adapter_id="pg-it", tables={"Instrument": "loka_it_instrument"})
    rows = _collect(adapter, TypedPredicate("Instrument", filters={"kind": "SovereignBond"}))
    assert [r["id"] for r in rows] == ["US10Y"]
