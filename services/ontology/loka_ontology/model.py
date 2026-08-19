"""Ontology meta-schema — Ω = (E, V, R, ⪯, C, A, N).

This is the core: it defines *how any ontology is represented*, independent of domain.
The structures here are the empty machinery; concrete domain entities/relations (the
content) are loaded from a YAML definition file once a vertical domain is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class VerbClass(StrEnum):
    """Act-class partition of the verb vocabulary: V = Vfact ⊔ Vcomm ⊔ Vinst."""

    FACTUAL = "factual"  # Vfact: acts on the objective world (TRADE, INVEST, HEDGE...)
    COMMUNICATIVE = "communicative"  # Vcomm: speech acts (ANNOUNCE, FORECAST...)
    INSTITUTIONAL = "institutional"  # Vinst: change norms/permissions (REGULATE, VOTE...)


class BaseType(StrEnum):
    """Property value types (a small, portable subset of the Palantir base-type set)."""

    STRING = "string"
    INTEGER = "integer"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"


@dataclass(frozen=True)
class Property:
    """An attribute of an entity type, with a value type. Properties are inherited via ⪯."""

    name: str
    base_type: BaseType
    required: bool = False
    description: str | None = None


@dataclass(frozen=True)
class EntityType:
    """E: one entity type; ``subtype_of`` encodes the subtype order ⪯.

    e.g. SovereignBond.subtype_of = "Bond", Bond.subtype_of = "Instrument"
    → SovereignBond ⪯ Bond ⪯ Instrument.

    Properties declared here are inherited by subtypes; a subtype may override a property by
    redeclaring it under the same name.
    """

    name: str
    subtype_of: str | None = None
    properties: tuple[Property, ...] = ()
    # the adapter/table this object type is derived from (Palantir-style)
    backing: str | None = None


@dataclass(frozen=True)
class Verb:
    """V: one verb and its act class."""

    name: str
    verb_class: VerbClass


class Cardinality(StrEnum):
    """Link-type cardinality (from → to)."""

    ONE_TO_ONE = "one_to_one"  # each from ≤ 1 to, each to ≤ 1 from
    ONE_TO_MANY = "one_to_many"  # from is the "one" side: each to ≤ 1 from
    MANY_TO_ONE = "many_to_one"  # to is the "one" side: each from ≤ 1 to
    MANY_TO_MANY = "many_to_many"  # unconstrained


@dataclass(frozen=True)
class Relation:
    """R: a directed relation between entity types, e.g. regulator-of(Regulator → Instrument).

    ``via`` names the field on the ``from_type`` that carries the link — how the relation is
    actually traversed in data. Declaring it in Ω rather than inferring it from a naming
    convention keeps the whole traversal ontology-authorized: a multi-hop path is derived from
    the declared relations *and* walked with the declared keys, so changing the ontology changes
    both. A relation without ``via`` is a type-level statement that cannot be traversed.
    """

    name: str
    from_type: str
    to_type: str
    # ``None`` means the ontology did not state a cardinality — which is not the same as stating
    # the unconstrained one. Review needs that difference: an unconfirmed link is a question,
    # a declared many-to-many is an answer. Use ``effective_cardinality`` when enforcing.
    cardinality: Cardinality | None = None
    via: str | None = None

    @property
    def effective_cardinality(self) -> Cardinality:
        """The cardinality to enforce: the declared one, or unconstrained when none was given."""
        return self.cardinality or Cardinality.MANY_TO_MANY


@dataclass(frozen=True)
class TypingConstraint:
    """One member of C: which agent and target types a verb may be used with.

    e.g. REGULATE requires agent ⪯ Regulator and target ⪯ Instrument ∨ ⪯ PolicyLever.

    C is part of Ω and constrains what may be *done*. It is distinct from CΩ, the load-time
    rules that decide whether an Ω is admissible at all — those live in the loader. One is
    inside the object; the other judges it.
    """

    verb: str
    agent_must_be: str  # agent must be ⪯ this type
    target_must_be: tuple[str, ...]  # target must be ⪯ one of these


@dataclass(frozen=True)
class ActionType:
    """An action the agent can perform — Ω's 4th primitive: a guard and an effect.

    ``verb`` names the action verb (must be a Verb in V); ``target`` the entity type it acts on;
    ``guard`` a precondition that must hold in the world state before it may run; ``effect`` the
    state change it produces. The declarative guard/effect strings are the starting point; the
    action layer evaluates them and gates execution.

    ``controllable`` splits the action set A = Au ⊎ Ac. A controllable action is one this system
    performs; an uncontrollable one is a change the environment makes, modelled so that its
    effect can be reasoned about but never proposed. The split is what makes norms well-formed:
    a norm answers "what must/may/must not be *done*", and there is no one to address that to
    for a change nobody chose.
    """

    name: str
    verb: str
    target: str
    guard: str = ""
    effect: str = ""
    controllable: bool = True


class NormStatus(StrEnum):
    """Deontic status of a controllable action in a state: N(s, ac)."""

    PERMITTED = "permitted"  # may be done — the default when no norm speaks
    MANDATORY = "mandatory"  # must be done; NOT doing it is itself a violation
    FORBIDDEN = "forbidden"  # must not be done


@dataclass(frozen=True)
class Norm:
    """One norm: in states satisfying ``when``, this action has this deontic status.

    Separate from an action's ``guard``, and the distinction is the whole point. A guard says
    whether an action *can* run — a precondition of the mechanism. A norm says whether it *may*,
    *must*, or *must not* — an obligation, which holds whether or not anyone honours it. An
    unsatisfied guard means the action is not available; a violated norm means something is
    wrong. Collapsing them loses the ability to say a system did something it was not allowed to
    do, and — the case a two-valued permitted/forbidden model cannot state at all — that a system
    failed to do something it was obliged to do.

    ``when`` empty means the norm holds unconditionally.
    """

    name: str
    action: str  # the ActionType this norm governs
    status: NormStatus
    when: str = ""
    rationale: str = ""


@dataclass
class Ontology:
    """A complete ontology Ω = (E, V, R, ⪯, C, A, N).

    C here is the typing-constraint set — which entity types may be agent and target of a verb.
    It is not CΩ: that name belongs to the load-time rules *about* an Ω, which live in the
    loader and decide whether an Ω is admissible at all. One is part of the object, the other
    judges the object, and using one symbol for both made a sentence about either ambiguous.

    The subtype order ⪯ is encoded in each EntityType's ``subtype_of`` field. ``actions`` (A) is
    the action-type set — the 4th primitive — partitioned into Au ⊎ Ac by ``controllable``.
    ``norms`` (N) assigns each controllable action a deontic status per state.
    """

    version: str
    entities: dict[str, EntityType] = field(default_factory=dict)
    verbs: dict[str, Verb] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    constraints: list[TypingConstraint] = field(default_factory=list)
    actions: list[ActionType] = field(default_factory=list)
    norms: list[Norm] = field(default_factory=list)
