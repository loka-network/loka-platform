"""Ontology lifecycle — draft → validated → published.

A generated ontology is a *proposal*, not an authority. `CΩ` catches structural faults (undefined
references, cycles, incompatible subtype overrides), but it cannot catch a proposal that is
structurally sound and semantically wrong: a numeric field typed `string`, a relation pointing the
wrong way, a missing cardinality, an absent guard. Those need a human.

So the lifecycle makes the human step part of the architecture rather than a convention:

    draft       a builder's proposal. May be edited freely. Cannot authorize an answer.
    validated   passed CΩ. Still cannot authorize an answer — no one has approved it.
    published   a human approved it. Frozen, given a version, and the *only* state a method may
                bind to or an audit record may cite.

This is what answers "how do you know the generated ontology is right?" — the answer is not that
the model is accurate, but that its output cannot reach a decision without passing CΩ and a
reviewer. :func:`review_checklist` makes that review concrete by naming what a builder cannot
know: the items below are exactly the ones absent from, or unreliable in, the source text.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DRAFT = "draft"
VALIDATED = "validated"
PUBLISHED = "published"

# Property names that almost always denote a quantity — a builder typing these as `string`
# is the single most common defect in a generated ontology.
_NUMERIC_HINTS = (
    "rate", "gdp", "count", "pct", "percent", "share", "amount", "price", "value", "index",
    "ratio", "per_capita", "mortality", "population", "income", "cost", "spend", "exp",
    "total", "avg", "average", "score", "level", "age", "year",
)

#: Beyond this many findings of one kind, they are reported as one item naming all of them.
#: A reviewer reading 62 separate lines saying "this relation declares no cardinality" is not
#: better informed than one reading a single line saying 31 of them do not, and is likelier to
#: stop reading — at which point the specific findings underneath are lost too.
_AGGREGATE_AT = 5


def _looks_numeric(name: str) -> bool:
    """Whether an attribute name suggests a quantity.

    Matched on whole words, not substrings. "country" contains "count", and a checklist that
    tells a reviewer their country field is probably a number teaches them the checklist is
    unreliable — which costs more than the finding was worth.
    """
    lowered = name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    return any(h in tokens if "_" not in h else h in lowered for h in _NUMERIC_HINTS)


class OntologyStateError(RuntimeError):
    """An operation was attempted from a state that does not allow it."""


@dataclass
class OntologyRecord:
    """One ontology through its lifecycle."""

    ontology_id: str
    yaml: str
    state: str = DRAFT
    version: str | None = None          # assigned at publish; None until then
    source: str = "builder"             # who proposed it
    review: list[dict[str, Any]] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    #: How this proposal was produced, in enough detail to reproduce it: the model, the prompt
    #: it was given verbatim, and a digest of the input. Without it, two drafts of the same
    #: domain that differ because a different model or a different prompt made them are
    #: indistinguishable in the record — both say "built by an LLM" — and the only honest answer
    #: to "why does yours have six concepts and mine nine" becomes "we don't know".
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_yaml: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ontology_id": self.ontology_id,
            "state": self.state,
            "version": self.version,
            "source": self.source,
            "provenance": dict(self.provenance),
            "review": self.review,
            "history": list(self.history),
            "can_authorize_answers": self.state == PUBLISHED,
        }
        if include_yaml:
            d["ontology_yaml"] = self.yaml
        return d


def review_checklist(yaml_text: str) -> list[dict[str, Any]]:
    """What a human must decide about this ontology, itemised.

    Every item below is something a builder reading domain text *cannot* reliably determine —
    either the text does not contain it (cardinality, guards, units) or it describes the world
    without committing to the modelling choice (which attribute is an outcome vs a control).
    An empty list means only that the automatic checks found nothing, not that review is optional.
    """
    from loka_ontology import load_ontology_str

    items: list[dict[str, Any]] = []

    def add(kind: str, target: str, detail: str) -> None:
        items.append({"kind": kind, "target": target, "detail": detail})

    def add_grouped(
        kind: str,
        targets: list[str],
        one: Callable[[str], str],
        many: Callable[[int], str],
    ) -> None:
        """Itemise a few findings; summarise a lot of them, naming every target either way.

        A systematic gap and a specific oversight need different treatment. Thirty-one relations
        with no cardinality is one fact about how extraction works, not thirty-one discoveries,
        and listing it thirty-one times buries the handful of findings that are specific.
        """
        if not targets:
            return
        if len(targets) <= _AGGREGATE_AT:
            for t in targets:
                add(kind, t, one(t))
            return
        items.append({
            "kind": kind,
            "target": f"{len(targets)} of them",
            "detail": many(len(targets)),
            "targets": list(targets),  # nothing is dropped, only folded
            "count": len(targets),
        })

    try:
        onto = load_ontology_str(yaml_text)
    except Exception as exc:  # noqa: BLE001 - an unloadable draft is itself the finding
        add("does_not_load", "-", f"CΩ rejects this draft: {exc}")
        return items

    undescribed: list[str] = []
    no_cardinality: list[str] = []
    no_via: list[str] = []

    for ent in onto.entities.values():
        for p in ent.properties:
            name = p.name.lower()
            if p.base_type == "string" and _looks_numeric(name):
                add(
                    "suspect_base_type",
                    f"{ent.name}.{p.name}",
                    "typed 'string' but the name suggests a quantity; confirm the base type",
                )
            # Units belong to quantities. Asking which currency a boolean is denominated in
            # is noise, and it was being asked of every attribute of every type.
            if not p.description and p.base_type in ("double", "integer"):
                undescribed.append(f"{ent.name}.{p.name}")

    for rel in onto.relations:
        if rel.cardinality is None:  # undeclared, not "declared as unconstrained"
            no_cardinality.append(f"{rel.name} ({rel.from_type} -> {rel.to_type})")
        if rel.via is None:
            no_via.append(rel.name)

    add_grouped(
        "missing_units",
        undescribed,
        lambda t: (
            f"{t} is a quantity with no description; state its unit and basis"
        ),
        lambda n: (
            f"{n} numeric attributes carry no description. A quantity without a stated unit "
            "and basis is not usable: two sources will fill it with different things and "
            "nothing downstream can tell"
        ),
    )
    add_grouped(
        "undeclared_cardinality",
        no_cardinality,
        lambda t: (
            f"{t} states no cardinality; confirm whether either side is single-valued "
            "(source text rarely says)"
        ),
        lambda n: (
            f"{n} relations state no cardinality. A text describes what relates to what and "
            "almost never how many, so this is expected from extraction and has to be decided"
        ),
    )
    add_grouped(
        "undeclared_link_field",
        no_via,
        lambda t: (
            f"{t} has no 'via' field: the relation says the types are related but not how to "
            "follow the link, so no query can traverse it"
        ),
        lambda n: (
            f"{n} relations declare no 'via' field, so none of them can be traversed. The "
            "field a link is walked by is in the data, not in the prose the ontology came from"
        ),
    )

    if not onto.constraints:
        add(
            "no_constraints",
            "-",
            "no typing constraints declared; domain rules (e.g. which entity types a verb may "
            "act on) come from expertise, not from the source text",
        )

    if not onto.actions:
        add(
            "no_actions",
            "-",
            "no action types declared; an ontology without actions cannot govern anything",
        )
    for a in onto.actions:
        if not a.guard:
            add("missing_guard", a.name, "no precondition — a guard is a governance decision")
        if not a.effect:
            add("missing_effect", a.name, "no declared effect on the world state")

    # Method roles are not part of Ω: which attribute is the outcome, which is the policy dial
    # and which are controls is a modelling commitment the text cannot supply.
    add(
        "assign_method_roles",
        "-",
        "designate which attributes are outcome / policy dial / controls; the source text "
        "describes the world but does not assign causal roles",
    )

    # Overlapping names can mean one concept extracted twice — but a subtype is *supposed* to
    # carry its supertype's name (BulkyProduct ⪯ Product), so a declared ⪯ pair is not a finding.
    def related_by_subtyping(a: str, b: str) -> bool:
        chain_a, chain_b = {a}, {b}
        cur: str | None = a
        while cur is not None:
            chain_a.add(cur)
            cur = onto.entities[cur].subtype_of
        cur = b
        while cur is not None:
            chain_b.add(cur)
            cur = onto.entities[cur].subtype_of
        return b in chain_a or a in chain_b

    names = sorted(onto.entities)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if related_by_subtyping(first, second):
                continue
            if first.lower() in second.lower() or second.lower() in first.lower():
                add(
                    "possible_synonyms",
                    f"{first} / {second}",
                    "names overlap and neither is a declared subtype of the other; confirm these "
                    "are distinct types and not one concept extracted twice",
                )
    return items


class OntologyStore:
    """In-memory lifecycle store. One process, one store; persistence is a later concern."""

    def __init__(self) -> None:
        self._records: dict[str, OntologyRecord] = {}

    def create_draft(
        self,
        yaml_text: str,
        *,
        source: str = "builder",
        provenance: dict[str, Any] | None = None,
    ) -> OntologyRecord:
        """Register a proposal. It is a draft: editable, and unable to authorize anything.

        ``provenance`` records what produced it — model, prompt, input digest — so the draft can
        be reproduced and two drafts of one domain can be told apart by more than their content.
        """
        rec = OntologyRecord(
            ontology_id=uuid.uuid4().hex[:12],
            yaml=yaml_text,
            state=DRAFT,
            source=source,
            provenance=dict(provenance or {}),
            review=review_checklist(yaml_text),
            history=[f"created as draft (source={source})"],
        )
        self._records[rec.ontology_id] = rec
        return rec

    def get(self, ontology_id: str) -> OntologyRecord | None:
        return self._records.get(ontology_id)

    def list(self) -> list[dict[str, Any]]:
        return [r.as_dict(include_yaml=False) for r in self._records.values()]

    def update(self, ontology_id: str, yaml_text: str) -> OntologyRecord:
        """Accept a reviewer's edit. Runs CΩ: on success the record becomes ``validated``.

        A published ontology is frozen — editing it means publishing a new version, so that a
        decision already cited against a version keeps meaning what it meant.
        """
        from loka_ontology import load_ontology_str

        rec = self._records[ontology_id]
        if rec.state == PUBLISHED:
            raise OntologyStateError(
                f"ontology {ontology_id} is published (version {rec.version}) and frozen; "
                "publish a new version instead of editing it"
            )
        load_ontology_str(yaml_text)  # CΩ — raises OntologyLoadError naming the rule that failed
        rec.yaml = yaml_text
        rec.state = VALIDATED
        rec.review = review_checklist(yaml_text)
        rec.history.append("edited and passed CΩ -> validated")
        return rec

    def publish(self, ontology_id: str, version: str) -> OntologyRecord:
        """A human approves the ontology. Only from ``validated``, and only once per version."""
        from loka_ontology import load_ontology_str

        rec = self._records[ontology_id]
        if rec.state == PUBLISHED:
            raise OntologyStateError(
                f"ontology {ontology_id} is already published as {rec.version}"
            )
        if rec.state != VALIDATED:
            raise OntologyStateError(
                f"ontology {ontology_id} is {rec.state}; it must pass CΩ (PUT the reviewed YAML) "
                "before it can be published"
            )
        onto = load_ontology_str(rec.yaml)
        if onto.version != version:  # the version in the YAML is the one decisions will cite
            rec.yaml = rec.yaml.replace(f"version: {onto.version}", f"version: {version}", 1)
            load_ontology_str(rec.yaml)
        rec.state = PUBLISHED
        rec.version = version
        rec.history.append(f"approved and published as {version} (frozen)")
        return rec
