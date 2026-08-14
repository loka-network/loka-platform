"""Infer the skeleton of Ω from several tables at once — keys, links, and the subtype order.

Single-table inference reads one table and proposes one entity type. That leaves out the part of
Ω a single table cannot contain: how the tables connect. Relations, their traversal keys, their
cardinalities and the subtype order all live *between* tables, so they are invisible to a reader
of one, and they are the part that makes Ω more than a schema.

What makes this worth doing separately from asking a model is that the answers are decidable
from the values. A relation is proposed because one column's values are contained in another's
primary key — a fact about the data, checked by counting, which cannot be hallucinated. Every
proposal here therefore carries its evidence: how many rows matched, out of how many, how many
distinct values, and what the name similarity was. A reviewer can disagree with a conclusion;
they should never have to guess what it was based on.

What this CANNOT infer, and does not try to:

  * **Verbs and their act classes.** Nothing in a table says SUSPEND_SELLER exists, let alone
    that it is institutional rather than factual.
  * **Typing constraints C.** Which entity type may perform a verb is a rule about authority.
  * **Actions, guards, effects.** A weight limit for standard shipping is a business decision;
    the weights in the data are consistent with any limit.
  * **Norms N.** Whether disclosing a delay is obliged or merely allowed is not a property of
    the delays.
  * **What a relation MEANS.** The data shows that order lines carry an order id. That the
    relation is called "contains" and is read from the order's side is a modelling choice.

Those are exactly what review, and text-based extraction, are for. The two directions are
complementary rather than competing: this one gets the skeleton right and stays silent about
meaning; a model reading prose supplies meaning and cannot check the skeleton.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .infer import guess_primary_key, infer_entity_type
from .model import Cardinality, EntityType, Ontology, Relation

Row = Mapping[str, object]
Tables = Mapping[str, Sequence[Row]]

#: A link is proposed only when nearly every value resolves. Real data has orphans — a deleted
#: row, a late-arriving fact — so demanding perfection would reject true keys; accepting a
#: half-match would invent one out of a coincidence.
MIN_INCLUSION = 0.95

#: Below this many distinct values, containment stops being evidence. A column holding three
#: status codes is contained in plenty of things by accident.
MIN_DISTINCT = 5

#: A subtype's key set must sit almost entirely inside its supertype's.
MIN_SUBTYPE_INCLUSION = 0.99

#: Every column is tried against every key, so most candidates resolve for none of their values.
#: Those are unrelated columns, not rejected findings, and listing them buries the few
#: rejections a reviewer should actually look at. Below this, a candidate is counted but not
#: reported; the count is kept so nothing is dropped silently.
NOTABLE_REJECTION = 0.5


@dataclass(frozen=True)
class LinkEvidence:
    """One proposed relation and the counts behind it."""

    from_type: str
    from_column: str
    to_type: str
    to_column: str
    matched: int  # source rows whose value exists in the target key
    total: int  # source rows with a value at all
    distinct_values: int
    unique_in_source: bool
    cardinality: str
    name_match: str  # exact | suffix | unrelated
    accepted: bool
    reason: str

    @property
    def inclusion(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "from": f"{self.from_type}.{self.from_column}",
            "to": f"{self.to_type}.{self.to_column}",
            "matched": self.matched,
            "total": self.total,
            "inclusion": round(self.inclusion, 4),
            "distinct_values": self.distinct_values,
            "cardinality": self.cardinality,
            "name_match": self.name_match,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SubtypeEvidence:
    """One proposed ⪯ edge and the counts behind it."""

    subtype: str
    supertype: str
    key_matched: int
    key_total: int
    subtype_rows: int
    supertype_rows: int
    extra_columns: tuple[str, ...]
    accepted: bool
    reason: str

    @property
    def inclusion(self) -> float:
        return self.key_matched / self.key_total if self.key_total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "subtype": self.subtype,
            "supertype": self.supertype,
            "key_inclusion": round(self.inclusion, 4),
            "rows": f"{self.subtype_rows} of {self.supertype_rows}",
            "extra_columns": list(self.extra_columns),
            "accepted": self.accepted,
            "reason": self.reason,
        }


#: The parts of Ω that are decisions rather than observations. Reported with every inference so
#: a draft is never mistaken for a finished ontology.
NOT_INFERABLE = (
    "verbs (V) and their act classes — nothing in a table names an action",
    "typing constraints (C) — which type may perform a verb is a rule about authority",
    "actions, guards and effects — a threshold is a business decision, not a property of values",
    "norms (N) — whether an action is obliged or merely allowed is not in the data",
    "relation names and orientation — 'contains' read from the order's side is a modelling choice",
)


@dataclass(frozen=True)
class InferenceReport:
    """Everything proposed, everything rejected, and what was never attempted."""

    keys: dict[str, str | None] = field(default_factory=dict)
    links: tuple[LinkEvidence, ...] = ()
    subtypes: tuple[SubtypeEvidence, ...] = ()
    #: column/key pairs tried and dismissed without reaching NOTABLE_REJECTION
    unrelated_pairs: int = 0
    not_inferable: tuple[str, ...] = NOT_INFERABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_keys": dict(self.keys),
            "links": [link.as_dict() for link in self.links],
            "subtypes": [s.as_dict() for s in self.subtypes],
            "accepted": {
                "links": sum(1 for x in self.links if x.accepted),
                "subtypes": sum(1 for x in self.subtypes if x.accepted),
            },
            "rejected": {
                "links": sum(1 for x in self.links if not x.accepted),
                "subtypes": sum(1 for x in self.subtypes if not x.accepted),
            },
            "unrelated_pairs_dismissed": self.unrelated_pairs,
            "not_inferable": list(self.not_inferable),
        }


def _values(rows: Sequence[Row], column: str) -> list[object]:
    return [r[column] for r in rows if r.get(column) not in (None, "")]


def _name_match(source_column: str, target_column: str, target_type: str) -> str:
    a, b, t = source_column.lower(), target_column.lower(), target_type.lower()
    if a == b:
        return "exact"
    if a.endswith(b) or b.endswith(a) or a.startswith(t):
        return "suffix"
    return "unrelated"


def _detect_subtypes(tables: Tables, keys: Mapping[str, str | None]) -> list[SubtypeEvidence]:
    """A table whose key values sit inside another's, with no column the other lacks, is that
    other's subtype: every one of its rows *is* one of the other's rows, described further.

    Requiring the column set to be a subset is what keeps this apart from an ordinary link. Two
    tables sharing a key but describing different things — an order and its shipment — each
    carry columns the other does not, and the extras are reported so a reviewer can see why the
    pair was or was not read as ⪯.
    """
    out: list[SubtypeEvidence] = []
    for sub, sub_rows in tables.items():
        sub_key = keys.get(sub)
        if not sub_key or not sub_rows:
            continue
        sub_cols = {c for r in sub_rows for c in r}
        sub_values = {repr(v) for v in _values(sub_rows, sub_key)}
        if not sub_values:
            continue
        for sup, sup_rows in tables.items():
            if sup == sub or keys.get(sup) != sub_key or not sup_rows:
                continue
            sup_values = {repr(v) for v in _values(sup_rows, sub_key)}
            matched = len(sub_values & sup_values)
            extra = tuple(sorted(sub_cols - {c for r in sup_rows for c in r}))
            inclusion = matched / len(sub_values)

            if len(sub_rows) >= len(sup_rows):
                reason = "not smaller than the candidate supertype"
                accepted = False
            elif inclusion < MIN_SUBTYPE_INCLUSION:
                reason = f"only {inclusion:.1%} of keys are present in {sup}"
                accepted = False
            elif extra:
                reason = f"carries columns {sup} does not: {', '.join(extra)}"
                accepted = False
            else:
                reason = (
                    f"every key is a {sup} key and no column is new — each row is a {sup}"
                )
                accepted = True

            out.append(
                SubtypeEvidence(
                    subtype=sub,
                    supertype=sup,
                    key_matched=matched,
                    key_total=len(sub_values),
                    subtype_rows=len(sub_rows),
                    supertype_rows=len(sup_rows),
                    extra_columns=extra,
                    accepted=accepted,
                    reason=reason,
                )
            )
    return out


def _detect_links(
    tables: Tables, keys: Mapping[str, str | None], skip: set[tuple[str, str]]
) -> tuple[list[LinkEvidence], int]:
    """A column whose values are contained in another table's primary key is a foreign key.

    Only a *primary* key is a valid target. That single condition removes the false positive
    this method is most prone to: two tables carrying the same vocabulary in an ordinary column
    — a seller's state and a customer's state both holding Brazilian state codes — overlap
    completely without one referencing the other. Neither is a key, so neither is proposed.

    Low-cardinality columns are excluded for the same reason from the other side: containment in
    a set of three status codes is arithmetic, not evidence.
    """
    out: list[LinkEvidence] = []
    dismissed = 0
    key_sets: dict[str, set[str]] = {}
    for name, rows in tables.items():
        key = keys.get(name)
        # A table with no usable key cannot be the target of a foreign key: there is nothing for
        # a value to resolve to. It is still a source, so it stays in the loop below.
        key_sets[name] = {repr(v) for v in _values(rows, key)} if key else set()
    for source, rows in tables.items():
        columns = {c for r in rows for c in r}
        for column in sorted(columns):
            if column == keys.get(source):
                continue
            vals = _values(rows, column)
            if not vals:
                continue
            reprs = [repr(v) for v in vals]
            distinct = len(set(reprs))
            for target, target_key in keys.items():
                if target == source or not target_key or (source, target) in skip:
                    continue
                target_values = key_sets[target]
                if not target_values:
                    continue
                matched = sum(1 for v in reprs if v in target_values)
                inclusion = matched / len(reprs)
                name_match = _name_match(column, target_key, target)

                if inclusion < MIN_INCLUSION:
                    accepted, reason = False, f"only {inclusion:.1%} of values are {target} keys"
                elif distinct < MIN_DISTINCT:
                    accepted, reason = False, (
                        f"only {distinct} distinct values — containment is coincidence at this "
                        "cardinality"
                    )
                elif name_match == "unrelated":
                    accepted, reason = False, (
                        f"values resolve but {column!r} bears no relation to {target_key!r}"
                    )
                else:
                    accepted, reason = True, (
                        f"{matched}/{len(reprs)} values are {target}.{target_key}"
                    )

                if not accepted and inclusion < NOTABLE_REJECTION:
                    dismissed += 1
                    continue

                unique = distinct == len(reprs)
                out.append(
                    LinkEvidence(
                        from_type=source,
                        from_column=column,
                        to_type=target,
                        to_column=target_key,
                        matched=matched,
                        total=len(reprs),
                        distinct_values=distinct,
                        unique_in_source=unique,
                        # Each source row points at one target. Whether several rows point at the
                        # same one is what separates the two cases, and it is countable.
                        cardinality=(
                            Cardinality.ONE_TO_ONE.value if unique
                            else Cardinality.MANY_TO_ONE.value
                        ),
                        name_match=name_match,
                        accepted=accepted,
                        reason=reason,
                    )
                )
    return out, dismissed


def infer_ontology_from_tables(
    tables: Tables, *, version: str = "draft-v1", backing: Mapping[str, str] | None = None
) -> tuple[Ontology, InferenceReport]:
    """Derive a draft Ω from several tables, with the evidence for every proposal.

    Returns the ontology and a report. The report is not decoration: a proposal a reviewer
    cannot audit is a proposal they can only accept or reject on faith, and the whole reason to
    infer structure from values rather than ask a model is that the answer can be checked.
    """
    keys = {name: guess_primary_key(rows) for name, rows in tables.items()}
    subtypes = _detect_subtypes(tables, keys)

    # A subtype's key is contained in its supertype's key by definition, so the same pair would
    # also read as a foreign key. It is one fact, and ⪯ is the stronger reading of it.
    accepted_subtypes = {s.subtype: s.supertype for s in subtypes if s.accepted}
    skip = {(sub, sup) for sub, sup in accepted_subtypes.items()}
    links, dismissed = _detect_links(tables, keys, skip)

    entities: dict[str, EntityType] = {}
    for name, rows in tables.items():
        entities[name] = infer_entity_type(
            name,
            rows,
            subtype_of=accepted_subtypes.get(name),
            backing=(backing or {}).get(name, name),
        )

    relations = [
        Relation(
            # Naming is not inferable, so the name states the fact rather than dressing it up as
            # a domain term a reviewer would then have to un-learn.
            name=f"{link.from_type.lower()}_to_{link.to_type.lower()}",
            from_type=link.from_type,
            to_type=link.to_type,
            cardinality=Cardinality(link.cardinality),
            via=link.from_column,
        )
        for link in links
        if link.accepted
    ]

    onto = Ontology(version=version, entities=entities, relations=relations)
    report = InferenceReport(
        keys=dict(keys),
        links=tuple(links),
        subtypes=tuple(subtypes),
        unrelated_pairs=dismissed,
    )
    return onto, report
