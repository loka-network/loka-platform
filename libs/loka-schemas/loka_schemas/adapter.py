"""The data-access contract — the MemoryAdapter interface.

This is the only path enterprise data takes into the system. Two properties are contractual:
  - **read-only**: the interface exposes no write operation, so data cannot be mutated
    through it.
  - **scoped**: a session carries customer-granted OAuth scopes; a query outside scope is
    rejected at the boundary.

Concrete adapters (in-memory, SQL, document store, ...) live in the `adapters` service and
implement this Protocol. Consumers (state service, compiler) depend on this contract only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .data import TypedPredicate, TypedRow


class AdapterError(Exception):
    """Base class for adapter-boundary failures."""


class AuthenticationError(AdapterError):
    """Certificate could not be authenticated (mTLS / OAuth failure)."""


class ScopeError(AdapterError):
    """The session's scopes do not cover the requested entity type."""


@dataclass(frozen=True)
class Certificate:
    """Client credential presented at the boundary (mTLS identity + OAuth scopes, simplified)."""

    subject: str
    scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Session:
    """An authenticated, scope-bound session. Valid for the duration of a run."""

    subject: str
    scopes: frozenset[str]
    established_at: datetime


@runtime_checkable
class MemoryAdapter(Protocol):
    """Two-method read-only data adapter (doc §3.1.4).

    ``authenticate`` establishes a scoped session; ``query`` streams typed rows. There is no
    write method by design.
    """

    async def authenticate(self, cert: Certificate) -> Session: ...

    def query(
        self, predicate: TypedPredicate, session: Session
    ) -> AsyncIterator[TypedRow]: ...
