"""Which of the terms an extraction proposed are entity types, and which are not yet.

The test is not ours. Review of the previous draft gave it in three places:

  * *properties are attributes. But we also need the subtyping relation. vehicle has wheels
    (property), car is vehicle. So we have two relations: is and has.*
  * *For the verbs define a syntax:* ``verb(subject, object1, object2, complements..)``
  * *In a DB data is structured in fields for core concepts, e.g. client: name, address, email.*

Read together they define an entity type structurally: a kind of thing that **has** attributes
and can be the **subject** of a verb. Not a frequent noun, and not whatever a model is willing
to call a concept.

That distinction matters because the alternatives were tried and measured on one 872-word
document, and each one removed real concepts:

  a floor on how often a term is mentioned   removed Carrier, Marketplace, Platform, Contract
  a floor on how often a term occurs         removed Punctuality, ConsumerLaw, Courier, Flood
  asking the model to judge                  three runs, three different answers

Each of those asks whether a word is *important*, which is a judgement. This asks whether a term
carries structure in the ontology that was extracted, which is a fact about that ontology and
can be counted. It gives the same answer whichever model produced the draft — the property the
review asked for, established where it can actually hold.

On the run this was written against it takes 98 proposals to 21, and the 21 are the domain:
Seller, Shopper, Item, Purchase, Line, Parcel, Carrier, Courier, Marketplace, Delay,
FreightService, StandardService, Law, Obligation, Rule, Rate, Breach, Evidence, Basket, Team,
RegionalFlood. Noise, Handful, TightOne, Briefing, Middle, Figure, Matter and Door do not
survive it.

Nothing is discarded. What does not pass is carried as a candidate term, with the reason, and a
reviewer can promote any of them — the previous attempts at this were filters, and a filter that
drops a real concept leaves a reviewer nothing to notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Ontology


@dataclass
class TypePartition:
    """The proposals that carry structure, and the ones that do not, with reasons."""

    core: list[str] = field(default_factory=list)
    #: name -> why it did not qualify
    candidates: dict[str, str] = field(default_factory=dict)
    #: relations dropped because an endpoint is not an entity type
    dropped_relations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_types": sorted(self.core),
            "candidate_terms": dict(sorted(self.candidates.items())),
            "relations_not_between_types": sorted(self.dropped_relations),
        }


def partition_types(onto: Ontology) -> TypePartition:
    """Split proposed types into those that carry structure and those that do not.

    Subtyping closes the set in both directions: ``car is vehicle`` makes both meaningful even
    when one of them carries nothing else, because the ``is`` relation is itself structure. The
    closure runs to a fixed point so a chain is not half-admitted.
    """
    has_attributes = {e.name for e in onto.entities.values() if e.properties}
    subjects = {r.from_type for r in onto.relations}
    core = has_attributes & subjects

    while True:
        grown = set(core)
        for e in onto.entities.values():
            if e.subtype_of and (e.name in core or e.subtype_of in core):
                grown |= {e.name, e.subtype_of}
        if grown == core:
            break
        core = grown
    core &= set(onto.entities)

    candidates: dict[str, str] = {}
    for name in onto.entities:
        if name in core:
            continue
        if name in has_attributes:
            candidates[name] = "has attributes, but is the subject of no verb"
        elif name in subjects:
            candidates[name] = "is the subject of a verb, but has no attributes"
        else:
            candidates[name] = "has no attributes and is the subject of no verb"

    dropped = [
        f"{r.from_type} -{r.name}-> {r.to_type}"
        for r in onto.relations
        if r.from_type not in core or r.to_type not in core
    ]
    return TypePartition(core=sorted(core), candidates=candidates, dropped_relations=dropped)


def restrict(onto: Ontology, partition: TypePartition) -> Ontology:
    """The ontology with only the entity types, and only the relations between them.

    A relation whose endpoint is not an entity type cannot load — an endpoint has to be a
    declared type — so keeping one would mean keeping the candidate as a type, which is the
    thing being decided. They are recorded in the partition instead of being silently lost.
    """
    keep = set(partition.core)
    return Ontology(
        version=onto.version,
        entities={n: e for n, e in onto.entities.items() if n in keep},
        verbs=dict(onto.verbs),
        relations=[
            r for r in onto.relations if r.from_type in keep and r.to_type in keep
        ],
        constraints=[
            c
            for c in onto.constraints
            if c.agent_must_be in keep and all(t in keep for t in c.target_must_be)
        ],
        actions=[a for a in onto.actions if a.target in keep],
        norms=list(onto.norms),
    )
