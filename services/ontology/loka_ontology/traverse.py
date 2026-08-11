"""Walking an Ω-derived path over data.

:meth:`OntologyEngine.path_between` answers *which way to go*; this answers *what is there*. The
two are deliberately separate: the route comes from the declared relations, and following it uses
only the ``via`` fields those relations declare, so no join is written by hand anywhere. Change
the ontology and both the route and the fields used to walk it change with it.

A dataset here is simply ``{entity_type: [row, ...]}`` — the shape ``InMemoryAdapter`` already
uses. Rows of a subtype are visible when reading its supertype, since a BulkyProduct is a Product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .engine import OntologyEngine
from .model import Relation

Row = Mapping[str, Any]
Dataset = Mapping[str, Sequence[Row]]


class TraversalError(RuntimeError):
    """The path cannot be walked over this data."""


def rows_of_type(engine: OntologyEngine, dataset: Dataset, entity_type: str) -> list[Row]:
    """Every row that *is* an ``entity_type`` — its own rows plus those of its subtypes (⪯)."""
    rows: list[Row] = list(dataset.get(entity_type, []))
    for sub in engine.subtypes_of(entity_type):
        rows.extend(dataset.get(sub, []))
    return rows


def follow(
    engine: OntologyEngine,
    dataset: Dataset,
    path: Sequence[tuple[Relation, bool]],
    start: Sequence[Row],
    *,
    start_type: str,
) -> list[Row]:
    """Walk ``path`` from ``start``, returning the rows reached at the far end.

    Each step matches the relation's ``via`` field on both sides — the same field regardless of
    direction, which is why a relation can be followed backwards as readily as forwards. A step
    whose relation declares no ``via`` raises rather than silently guessing a key.
    """
    rows = list(start)
    for rel, forward in path:
        if not rel.via:
            raise TraversalError(
                f"relation {rel.name} declares no 'via' field, so it states that {rel.from_type} "
                f"and {rel.to_type} are related without saying how to follow the link"
            )
        nxt = rel.to_type if forward else rel.from_type
        keys = {r[rel.via] for r in rows if r.get(rel.via) is not None}
        rows = [r for r in rows_of_type(engine, dataset, nxt) if r.get(rel.via) in keys]
    return rows


def reach(
    engine: OntologyEngine,
    dataset: Dataset,
    *,
    from_type: str,
    to_type: str,
    start: Sequence[Row],
    max_hops: int = 4,
) -> dict[str, Any]:
    """Derive the route from ``from_type`` to ``to_type`` and walk it — the whole multi-hop answer.

    Returns the route (for display and audit), the rows reached, and whether reaching the target
    required narrowing to a subtype, which the caller must then check on the data.
    """
    path = engine.path_between(from_type, to_type, max_hops=max_hops)
    narrowed = False
    if path is None:
        path = engine.path_between(from_type, to_type, max_hops=max_hops, allow_narrowing=True)
        narrowed = path is not None
    if path is None:
        return {
            "from": from_type, "to": to_type, "route": None, "hops": None,
            "rows": [], "reason": f"the ontology declares no route from {from_type} to {to_type}",
        }
    reached = follow(engine, dataset, path, start, start_type=from_type)
    if narrowed:  # landed on a supertype: keep only rows that really are the target type
        target_ids = {id(r) for r in rows_of_type(engine, dataset, to_type)}
        reached = [r for r in reached if id(r) in target_ids]
    return {
        "from": from_type,
        "to": to_type,
        "route": [f"{rel.name}{'>' if fwd else '<'}(via {rel.via})" for rel, fwd in path],
        "hops": len(path),
        "requires_narrowing": narrowed,
        "rows": reached,
    }
