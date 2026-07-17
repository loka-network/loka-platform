"""Infer a draft ontology from data — the Palantir-style "dataset → object type" step.

Given sample rows (from an adapter or a table), this derives a draft object type: one property
per column with an inferred base type, a guessed primary key, and the backing table recorded.
The output is a *draft* meant for human curation (types/PK guesses may need adjustment) — the
same "derive then curate" model Palantir uses, minus the GUI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime

import yaml

from .model import BaseType, EntityType, Ontology, Property

Row = Mapping[str, object]


def infer_base_type(values: Sequence[object]) -> BaseType:
    """Infer a base type from a column's sample values (None ignored)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return BaseType.STRING

    def all_are(pred: Callable[[object], bool]) -> bool:
        return all(pred(v) for v in vals)

    if all_are(lambda v: isinstance(v, bool)):
        return BaseType.BOOLEAN
    if all_are(lambda v: isinstance(v, int) and not isinstance(v, bool)):
        return BaseType.INTEGER
    if all_are(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)):
        return BaseType.DOUBLE
    if all_are(lambda v: isinstance(v, datetime)):
        return BaseType.TIMESTAMP
    if all_are(lambda v: isinstance(v, date) and not isinstance(v, datetime)):
        return BaseType.DATE
    return BaseType.STRING


def _columns(rows: Sequence[Row]) -> list[str]:
    seen: dict[str, None] = {}  # preserve first-seen order
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def guess_primary_key(rows: Sequence[Row]) -> str | None:
    """Guess a primary key: present & non-null & unique across all rows.

    Prefers columns named like an id (``id`` / ``*_id`` / ``*_code``).
    """
    if not rows:
        return None
    candidates: list[str] = []
    for col in _columns(rows):
        values = [row.get(col) for row in rows]
        if any(v is None for v in values):
            continue
        if len({repr(v) for v in values}) == len(values):  # unique
            candidates.append(col)

    def rank(col: str) -> int:
        low = col.lower()
        if low == "id":
            return 0
        if low.endswith("_id") or low.endswith("_code"):
            return 1
        return 2

    return min(candidates, key=rank) if candidates else None


def infer_entity_type(
    name: str,
    rows: Sequence[Row],
    *,
    subtype_of: str | None = None,
    backing: str | None = None,
) -> EntityType:
    """Derive a draft object type from sample rows."""
    pk = guess_primary_key(rows)
    n = len(rows)
    props: list[Property] = []
    for col in _columns(rows):
        values = [row.get(col) for row in rows]
        base = infer_base_type(values)
        # required if present & non-null in every row, or if it's the guessed PK
        present_everywhere = n > 0 and all(row.get(col) is not None for row in rows)
        props.append(Property(name=col, base_type=base, required=present_everywhere or col == pk))
    return EntityType(name=name, subtype_of=subtype_of, properties=tuple(props), backing=backing)


def infer_ontology_from_rows(
    entity_type: str,
    rows: Sequence[Row],
    *,
    subtype_of: str | None = None,
    backing: str | None = None,
    version: str = "draft-v1",
) -> Ontology:
    """Build a one-entity draft ontology from sample rows."""
    et = infer_entity_type(entity_type, rows, subtype_of=subtype_of, backing=backing)
    return Ontology(version=version, entities={et.name: et})


def to_yaml(ontology: Ontology) -> str:
    """Serialise an ontology back to the loader's YAML format (round-trips through load)."""
    entities: list[dict[str, object]] = []
    for e in ontology.entities.values():
        item: dict[str, object] = {"type": e.name}
        if e.subtype_of is not None:
            item["subtype_of"] = e.subtype_of
        if e.backing is not None:
            item["backing"] = e.backing
        if e.properties:
            item["properties"] = [
                {"name": p.name, "type": p.base_type.value, "required": p.required}
                for p in e.properties
            ]
        entities.append(item)

    doc: dict[str, object] = {"version": ontology.version, "entities": entities}
    if ontology.verbs:
        doc["verbs"] = [
            {"name": v.name, "class": v.verb_class.value} for v in ontology.verbs.values()
        ]
    if ontology.relations:
        doc["relations"] = [
            {
                "name": r.name,
                "from": r.from_type,
                "to": r.to_type,
                "cardinality": r.cardinality.value,
            }
            for r in ontology.relations
        ]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
