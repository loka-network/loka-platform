"""Postgres-backed read-only adapter — a production data adapter.

Implements the same ``MemoryAdapter`` contract as the in-memory reference adapter, but reads
from a real Postgres warehouse. The connection is opened read-only, so a write can never leave
the boundary; queries are parameterised and identifiers are safely quoted (no SQL injection);
rows stream from a server-side cursor (not bulk-copied); every row carries a lineage tag.

Requires ``psycopg`` and a reachable Postgres (see infra/docker-compose.yml). Verify with the
gated integration tests on a live database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime

import psycopg
from loka_schemas.adapter import AuthenticationError, Certificate, ScopeError, Session
from loka_schemas.data import Lineage, TypedPredicate, TypedRow
from psycopg import sql
from psycopg.rows import dict_row

from .sql_planner import plan_select


class PostgresAdapter:
    """Read-only adapter over a Postgres database, keyed by entity type → table."""

    def __init__(
        self,
        dsn: str,
        *,
        adapter_id: str,
        source: str = "postgres",
        tables: Mapping[str, str] | None = None,
        timestamp_field: str = "ts",
    ) -> None:
        self._dsn = dsn
        self._adapter_id = adapter_id
        self._source = source
        self._tables = dict(tables or {})
        self._timestamp_field = timestamp_field

    async def authenticate(self, cert: Certificate) -> Session:
        if not cert.subject:
            raise AuthenticationError("certificate has no subject")
        return Session(
            subject=cert.subject,
            scopes=cert.scopes,
            established_at=datetime.now(UTC),
        )

    async def query(
        self, predicate: TypedPredicate, session: Session
    ) -> AsyncIterator[TypedRow]:
        if not ("*" in session.scopes or predicate.entity_type in session.scopes):
            raise ScopeError(
                f"session {session.subject} is not scoped for {predicate.entity_type}"
            )
        table = self._tables.get(predicate.entity_type, predicate.entity_type)
        statement, params = self._build_query(table, predicate)

        conn = await psycopg.AsyncConnection.connect(self._dsn)
        try:
            await conn.set_read_only(True)  # writes are rejected at the DB level
            async with conn.cursor(name="loka_stream", row_factory=dict_row) as cur:
                await cur.execute(statement, params)
                retrieved_at = datetime.now(UTC)
                async for row in cur:
                    yield TypedRow(
                        entity_type=predicate.entity_type,
                        values=dict(row),
                        lineage=Lineage(
                            source=self._source,
                            adapter_id=self._adapter_id,
                            retrieved_at=retrieved_at,
                        ),
                    )
        finally:
            await conn.close()

    def _build_query(
        self, table: str, predicate: TypedPredicate
    ) -> tuple[str | sql.Composed, list[object]]:
        """Assemble the read.

        When the caller supplies a column projection — which it does when Ω describes the entity
        — the whole statement is planned by :func:`plan_select`: every identifier is validated
        against a plain-identifier pattern, every operator comes from a fixed set, and every
        value is a bound parameter. Injection is not filtered for; it cannot be expressed.

        A predicate with no columns falls back to ``SELECT *``, which is the shape for a table no
        ontology describes. That path exists for un-modelled data and reads whatever the table
        has, so it is the weaker one.
        """
        ranges: list[tuple[str, str, object]] = []
        if predicate.time_range is not None:
            if predicate.time_range.start is not None:
                ranges.append((self._timestamp_field, ">=", predicate.time_range.start))
            if predicate.time_range.end is not None:
                ranges.append((self._timestamp_field, "<", predicate.time_range.end))

        if predicate.columns:
            return plan_select(
                table, list(predicate.columns), filters=predicate.filters, ranges=ranges
            )

        conditions: list[sql.Composable] = []
        params: list[object] = []
        for key, expected in predicate.filters.items():
            conditions.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(expected)
        for column, op_text, value in ranges:
            conditions.append(
                sql.SQL("{} " + op_text + " %s").format(sql.Identifier(column))  # noqa: S608
            )
            params.append(value)
        statement = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        if conditions:
            statement = statement + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
        return statement, params
