"""Shared data contracts — the typed values that flow between services.

These types are the wire between adapters, the state service, and the compiler. They live in
loka-schemas (not in any single service) so every service depends on the same definitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Interval:
    """A half-open time interval [start, end). ``None`` bounds mean unbounded on that side."""

    start: datetime | None = None
    end: datetime | None = None

    def contains(self, t: datetime) -> bool:
        if self.start is not None and t < self.start:
            return False
        if self.end is not None and t >= self.end:
            return False
        return True


@dataclass(frozen=True)
class TypedPredicate:
    """A typed query against one entity type: equality filters + optional time range.

    Mirrors the doc's ``memory.get(EntityType, filter, time_range)`` contract.
    """

    entity_type: str
    filters: Mapping[str, object] = field(default_factory=dict)
    time_range: Interval | None = None


@dataclass(frozen=True)
class Lineage:
    """Data-lineage tag carried by every row, so any downstream value can be traced back."""

    source: str  # upstream system the row came from
    adapter_id: str  # adapter that produced it
    retrieved_at: datetime


@dataclass(frozen=True)
class TypedRow:
    """One typed row returned by an adapter. Immutable; carries its lineage."""

    entity_type: str
    values: Mapping[str, object]
    lineage: Lineage
