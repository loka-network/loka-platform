"""In-memory reference adapter — a read-only MemoryAdapter over an in-memory dataset.

For tests and local development. Demonstrates the boundary guarantees:
  - read-only by construction (no write method exists),
  - scope enforcement (a query outside the session's scopes is rejected),
  - streaming (rows are yielded one at a time, not bulk-copied),
  - lineage (every row carries a lineage tag).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime

from loka_schemas.adapter import (
    AuthenticationError,
    Certificate,
    ScopeError,
    Session,
)
from loka_schemas.data import Lineage, TypedPredicate, TypedRow

Dataset = Mapping[str, Sequence[Mapping[str, object]]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class InMemoryAdapter:
    """Read-only adapter backed by an in-memory dataset keyed by entity type."""

    def __init__(
        self,
        adapter_id: str,
        dataset: Dataset,
        *,
        source: str = "in-memory",
        timestamp_field: str = "ts",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._adapter_id = adapter_id
        self._dataset = dataset
        self._source = source
        self._timestamp_field = timestamp_field
        self._now = now or _utc_now

    async def authenticate(self, cert: Certificate) -> Session:
        if not cert.subject:
            raise AuthenticationError("certificate has no subject")
        return Session(
            subject=cert.subject,
            scopes=cert.scopes,
            established_at=self._now(),
        )

    async def query(
        self, predicate: TypedPredicate, session: Session
    ) -> AsyncIterator[TypedRow]:
        if not self._in_scope(predicate.entity_type, session):
            raise ScopeError(
                f"session {session.subject} is not scoped for {predicate.entity_type}"
            )
        for row in self._dataset.get(predicate.entity_type, []):
            if not self._matches(row, predicate):
                continue
            yield TypedRow(
                entity_type=predicate.entity_type,
                values=dict(row),  # copy: callers cannot mutate the source
                lineage=Lineage(
                    source=self._source,
                    adapter_id=self._adapter_id,
                    retrieved_at=self._now(),
                ),
            )

    @staticmethod
    def _in_scope(entity_type: str, session: Session) -> bool:
        return "*" in session.scopes or entity_type in session.scopes

    def _matches(self, row: Mapping[str, object], predicate: TypedPredicate) -> bool:
        for key, expected in predicate.filters.items():
            if row.get(key) != expected:
                return False
        if predicate.time_range is not None:
            ts = row.get(self._timestamp_field)
            if not isinstance(ts, datetime) or not predicate.time_range.contains(ts):
                return False
        return True
