"""Compare two ontologies over the same domain, and say what each has that the other lacks.

Two things need this, and they are the same operation.

**Cross-checking the two ways an ontology is built.** A model reading prose proposes concepts; a
reader of tables proposes structure. Where they agree, a proposal has two independent supports.
Where the prose has an entity the data cannot ground, that entity is a hypothesis — it may be
perfectly real and simply absent from the tables at hand, but it is not the same kind of claim as
one backed by rows, and a reviewer should be told which is which. Where the data has a link the
prose never mentions, the domain description is incomplete.

**Measuring what review actually changed.** Comparing a generated draft against the published
ontology it eventually became turns "a human reviewed it" into a list: these were added, these
were dropped, this cardinality was changed. Without that, review is a step everyone agrees is
important and nobody can point at.

The comparison is deliberately shallow: names, and for relations the pair of endpoints. It does
not try to decide that ``sold_by`` and ``fulfilled_by`` are the same relation under two names.
Guessing at synonymy would produce a comparison whose disagreements are as likely to be the
comparison's fault as the ontology's, and the point of this is to be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Ontology


@dataclass(frozen=True)
class OntologyComparison:
    """What the two ontologies share and where they differ. ``a`` is the reference."""

    entities_shared: tuple[str, ...]
    entities_only_in_a: tuple[str, ...]
    entities_only_in_b: tuple[str, ...]

    #: relations matched on (from_type, to_type), regardless of name
    edges_shared: tuple[str, ...]
    edges_only_in_a: tuple[str, ...]
    edges_only_in_b: tuple[str, ...]

    subtypes_shared: tuple[str, ...]
    subtypes_only_in_a: tuple[str, ...]
    subtypes_only_in_b: tuple[str, ...]

    #: an edge present in both but read in opposite directions — the same fact about the data,
    #: modelled from the other end. Not a disagreement about what is true.
    edges_reversed: tuple[str, ...]

    #: (edge, cardinality in a, cardinality in b)
    cardinality_differs: tuple[tuple[str, str, str], ...]

    #: parts of Ω only one side can carry at all, counted rather than compared
    verbs: tuple[int, int]
    constraints: tuple[int, int]
    actions: tuple[int, int]
    norms: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "entities": {
                "shared": list(self.entities_shared),
                "only_in_reference": list(self.entities_only_in_a),
                "only_in_candidate": list(self.entities_only_in_b),
            },
            "relations": {
                "shared": list(self.edges_shared),
                "only_in_reference": list(self.edges_only_in_a),
                "only_in_candidate": list(self.edges_only_in_b),
                "reversed": list(self.edges_reversed),
                "cardinality_differs": [
                    {"edge": e, "reference": x, "candidate": y}
                    for e, x, y in self.cardinality_differs
                ],
            },
            "subtypes": {
                "shared": list(self.subtypes_shared),
                "only_in_reference": list(self.subtypes_only_in_a),
                "only_in_candidate": list(self.subtypes_only_in_b),
            },
            "counts_only_one_side_can_hold": {
                "verbs": {"reference": self.verbs[0], "candidate": self.verbs[1]},
                "constraints": {
                    "reference": self.constraints[0], "candidate": self.constraints[1]
                },
                "actions": {"reference": self.actions[0], "candidate": self.actions[1]},
                "norms": {"reference": self.norms[0], "candidate": self.norms[1]},
            },
        }


def _edges(onto: Ontology) -> dict[tuple[str, str], str]:
    """Endpoint pair -> cardinality. Keyed on the pair, not the name, because naming is the part
    a machine cannot infer and a reviewer is free to choose."""
    return {
        (r.from_type, r.to_type): r.effective_cardinality.value for r in onto.relations
    }


def _subtype_edges(onto: Ontology) -> set[str]:
    return {
        f"{e.name} <= {e.subtype_of}"
        for e in onto.entities.values()
        if e.subtype_of is not None
    }


def compare_ontologies(reference: Ontology, candidate: Ontology) -> OntologyComparison:
    """Compare ``candidate`` against ``reference``. Neither is assumed to be correct."""
    ents_a, ents_b = set(reference.entities), set(candidate.entities)
    edges_a, edges_b = _edges(reference), _edges(candidate)
    subs_a, subs_b = _subtype_edges(reference), _subtype_edges(candidate)

    shared_edges = set(edges_a) & set(edges_b)
    # An edge only in one side, whose reverse is in the other, is the same edge modelled from
    # the other end. Reporting it as missing on both sides would be two false findings.
    only_a = {e for e in edges_a if e not in edges_b}
    only_b = {e for e in edges_b if e not in edges_a}
    reversed_pairs = {e for e in only_a if (e[1], e[0]) in only_b}
    only_a -= reversed_pairs
    only_b -= {(e[1], e[0]) for e in reversed_pairs}

    def label(e: tuple[str, str]) -> str:
        return f"{e[0]} -> {e[1]}"

    return OntologyComparison(
        entities_shared=tuple(sorted(ents_a & ents_b)),
        entities_only_in_a=tuple(sorted(ents_a - ents_b)),
        entities_only_in_b=tuple(sorted(ents_b - ents_a)),
        edges_shared=tuple(sorted(label(e) for e in shared_edges)),
        edges_only_in_a=tuple(sorted(label(e) for e in only_a)),
        edges_only_in_b=tuple(sorted(label(e) for e in only_b)),
        edges_reversed=tuple(sorted(label(e) for e in reversed_pairs)),
        cardinality_differs=tuple(
            sorted(
                (label(e), edges_a[e], edges_b[e])
                for e in shared_edges
                if edges_a[e] != edges_b[e]
            )
        ),
        subtypes_shared=tuple(sorted(subs_a & subs_b)),
        subtypes_only_in_a=tuple(sorted(subs_a - subs_b)),
        subtypes_only_in_b=tuple(sorted(subs_b - subs_a)),
        verbs=(len(reference.verbs), len(candidate.verbs)),
        constraints=(len(reference.constraints), len(candidate.constraints)),
        actions=(len(reference.actions), len(candidate.actions)),
        norms=(len(reference.norms), len(candidate.norms)),
    )


def grounding_checklist(text_built: Ontology, data_inferred: Ontology) -> list[dict[str, str]]:
    """Review items from cross-checking a text-built draft against what the data supports.

    This is the item the two-line design exists to produce. An entity a model read out of prose
    but which no table can ground is not thereby wrong — the tables in hand may simply not cover
    it — but it rests on a different kind of evidence, and shipping the two indistinguishably is
    how an ontology comes to contain things nobody can check.
    """
    cmp = compare_ontologies(data_inferred, text_built)
    items: list[dict[str, str]] = []
    for name in cmp.entities_only_in_b:
        items.append({
            "kind": "ungrounded_entity",
            "target": name,
            "detail": (
                f"{name} was read out of the domain text but no table supports it. Confirm the "
                "data source, or record it as a concept the current data cannot check."
            ),
        })
    for edge in cmp.edges_only_in_b:
        items.append({
            "kind": "ungrounded_relation",
            "target": edge,
            "detail": (
                f"{edge} is described in the text but no column's values resolve to the target's "
                "key. Either the link is carried elsewhere, or it does not hold in this data."
            ),
        })
    for name in cmp.entities_only_in_a:
        items.append({
            "kind": "undescribed_entity",
            "target": name,
            "detail": (
                f"the data contains {name}, which the domain text never mentions — the "
                "description is incomplete, or this table is out of scope."
            ),
        })
    for edge in cmp.edges_only_in_a:
        items.append({
            "kind": "undescribed_relation",
            "target": edge,
            "detail": f"the data supports {edge}, which the text does not describe.",
        })
    for edge, ref, cand in cmp.cardinality_differs:
        items.append({
            "kind": "cardinality_disagreement",
            "target": edge,
            "detail": (
                f"the data shows {ref}; the text was read as {cand}. One of them is about a "
                "sample and the other about the domain — decide which."
            ),
        })
    return items
