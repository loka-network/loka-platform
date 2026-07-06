"""Tests for the world-state service Eₜ."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from loka_schemas import Certificate, Lineage, Session, TypedPredicate, TypedRow
from loka_state import WorldState

T = datetime(2026, 1, 1, tzinfo=UTC)


class FakeAdapter:
    """Minimal read-only adapter for testing ingest (no cross-service dep)."""

    async def authenticate(self, cert: Certificate) -> Session:
        return Session(subject=cert.subject, scopes=cert.scopes, established_at=T)

    async def query(
        self, predicate: TypedPredicate, session: Session
    ) -> AsyncIterator[TypedRow]:
        rows = [
            {"id": "Fed", "rate": 0.05},
            {"id": "ECB", "rate": 0.04},
        ]
        for r in rows:
            yield TypedRow(
                entity_type=predicate.entity_type,
                values=r,
                lineage=Lineage(source="fake", adapter_id="a1", retrieved_at=T),
            )


def test_set_get_and_slice() -> None:
    st = WorldState()
    st.set("CentralBank.Fed.rate", 0.05, T)
    st.set("Instrument.US10Y.kind", "SovereignBond", T)
    assert st.get("CentralBank.Fed.rate") is not None
    sl = st.slice(["CentralBank"])
    assert sl == {"CentralBank.Fed.rate": 0.05}


def test_ingest_from_adapter() -> None:
    st = WorldState()
    n = asyncio.run(
        st.ingest_from(FakeAdapter(), TypedPredicate("CentralBank"), Session("t", frozenset(), T))
    )
    assert n == 2
    assert st.get("CentralBank.Fed.rate") is not None
    assert st.get("CentralBank.ECB.rate") is not None


def test_snapshot_hash_is_deterministic() -> None:
    a = WorldState()
    a.set("CentralBank.Fed.rate", 0.05, T)
    b = WorldState()
    b.set("CentralBank.Fed.rate", 0.05, T)
    assert a.snapshot_hash() == b.snapshot_hash()


def test_snapshot_hash_changes_with_state() -> None:
    a = WorldState()
    a.set("CentralBank.Fed.rate", 0.05, T)
    before = a.snapshot_hash()
    a.set("CentralBank.Fed.rate", 0.04, T)
    assert a.snapshot_hash() != before
