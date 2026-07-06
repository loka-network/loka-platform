"""World state Eₜ — a live, partially-observed mirror of the world right now.

Holds typed state-variable values with timestamps. Can ingest from read-only adapters (the
data-in path) and produce a deterministic snapshot hash for manifest pinning, so a compiled
W(q, t) can be reproduced exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from loka_schemas import MemoryAdapter, Session, TypedPredicate


@dataclass(frozen=True)
class StateValue:
    """One observed value with the time it was observed."""

    value: object
    ts: datetime


class WorldState:
    """Eₜ. Variable names follow ``<EntityType>.<id>.<field>``."""

    def __init__(self) -> None:
        self._vars: dict[str, StateValue] = {}

    def set(self, name: str, value: object, ts: datetime) -> None:
        self._vars[name] = StateValue(value, ts)

    def get(self, name: str) -> StateValue | None:
        return self._vars.get(name)

    def slice(self, entity_types: Iterable[str]) -> dict[str, object]:
        """Return the values whose variable name belongs to one of ``entity_types``."""
        prefixes = tuple(f"{e}." for e in entity_types)
        return {name: sv.value for name, sv in self._vars.items() if name.startswith(prefixes)}

    async def ingest_from(
        self,
        adapter: MemoryAdapter,
        predicate: TypedPredicate,
        session: Session,
        *,
        key_field: str = "id",
    ) -> int:
        """Pull rows from a read-only adapter into Eₜ. Returns the number of rows ingested."""
        count = 0
        async for row in adapter.query(predicate, session):
            rid = row.values.get(key_field, "?")
            for field_name, value in row.values.items():
                self.set(f"{row.entity_type}.{rid}.{field_name}", value, row.lineage.retrieved_at)
            count += 1
        return count

    def snapshot_hash(self) -> str:
        """Deterministic short hash of the current state (used as a manifest pin)."""
        items = sorted(
            (name, repr(sv.value), sv.ts.isoformat()) for name, sv in self._vars.items()
        )
        payload = "|".join(f"{name}={value}@{ts}" for name, value, ts in items)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
