"""Ports — the minimal interfaces consumers depend on, so implementations are swappable.

The compiler (and other consumers) depend on these Protocols rather than on concrete engine
classes. This is what lets an in-memory reference implementation be swapped for a real
backend (Neo4j causal graph, Postgres-backed state, ...) without touching consumer logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from .causal import CausalSlice


@runtime_checkable
class OntologyView(Protocol):
    """The read-only ontology surface the compiler needs."""

    @property
    def version(self) -> str: ...

    def has_entity(self, name: str) -> bool: ...


@runtime_checkable
class StateView(Protocol):
    """The read-only world-state surface the compiler needs."""

    def slice(self, entity_types: Iterable[str]) -> dict[str, object]: ...

    def snapshot_hash(self) -> str: ...


@runtime_checkable
class StateStore(StateView, Protocol):
    """A world state that can also be written to — what ingestion needs.

    The write surface is a separate protocol so that a consumer which only reads cannot write
    by accident: the compiler is handed a :class:`StateView` and therefore cannot mutate the
    state it is compiling against, while an ingestion path asks for this one. A concrete store
    satisfies both, so this costs nothing at the implementation side.
    """

    def set(self, name: str, value: object, ts: datetime) -> None: ...


@runtime_checkable
class CausalSlicer(Protocol):
    """The causal surface the compiler needs: produce Γ(q) for a set of targets."""

    def build_slice(self, targets: Sequence[str]) -> CausalSlice: ...
