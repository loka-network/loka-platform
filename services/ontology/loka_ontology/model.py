"""Ontology meta-schema — Ω = (E, V, R, ⪯, CΩ).

This is the core: it defines *how any ontology is represented*, independent of domain.
The structures here are the empty machinery; concrete domain entities/relations (the
content) are loaded from a YAML definition file once a vertical domain is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerbClass(str, Enum):
    """Act-class partition of the verb vocabulary: V = Vfact ⊔ Vcomm ⊔ Vinst."""

    FACTUAL = "factual"  # Vfact: acts on the objective world (TRADE, INVEST, HEDGE...)
    COMMUNICATIVE = "communicative"  # Vcomm: speech acts (ANNOUNCE, FORECAST...)
    INSTITUTIONAL = "institutional"  # Vinst: change norms/permissions (REGULATE, VOTE...)


@dataclass(frozen=True)
class EntityType:
    """E: one entity type; ``subtype_of`` encodes the subtype order ⪯.

    e.g. SovereignBond.subtype_of = "Bond", Bond.subtype_of = "Instrument"
    → SovereignBond ⪯ Bond ⪯ Instrument.
    """

    name: str
    subtype_of: str | None = None


@dataclass(frozen=True)
class Verb:
    """V: one verb and its act class."""

    name: str
    verb_class: VerbClass


@dataclass(frozen=True)
class Relation:
    """R: a directed relation between entity types, e.g. regulator-of(Regulator → Instrument)."""

    name: str
    from_type: str
    to_type: str


@dataclass(frozen=True)
class TypingConstraint:
    """One typing constraint in CΩ (simplified form).

    Constrains the agent/target types allowed for a verb.
    e.g. REGULATE requires agent ⪯ Regulator and target ⪯ Instrument ∨ ⪯ PolicyLever.
    Note: this declarative form is the starting point; the full CΩ (~250 rules) will be
    implemented with Soufflé/Datalog later.
    """

    verb: str
    agent_must_be: str  # agent must be ⪯ this type
    target_must_be: tuple[str, ...]  # target must be ⪯ one of these


@dataclass
class Ontology:
    """A complete ontology Ω = (E, V, R, ⪯, CΩ).

    The subtype order ⪯ is encoded in each EntityType's ``subtype_of`` field.
    """

    version: str
    entities: dict[str, EntityType] = field(default_factory=dict)
    verbs: dict[str, Verb] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    constraints: list[TypingConstraint] = field(default_factory=list)
