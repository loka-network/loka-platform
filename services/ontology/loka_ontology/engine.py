"""Ontology engine: queries and type checks over a loaded Ontology.

This is the public API the engine exposes to downstream services (grounding, compiler).
Downstream accesses the ontology only through these methods, never its internals — this is
what keeps the system decoupled (engineering principle #3).

The checks start as a simplified deterministic form; the full CΩ (~250 rules) will be
implemented with Soufflé/Datalog later.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

from .model import ActionType, BaseType, Cardinality, Ontology, Property, Relation, VerbClass


@dataclass(frozen=True)
class BindingCheck:
    """Result of one (verb, agent_type, target_type) binding check."""

    ok: bool
    rule: str | None = None  # matched constraint (when ok)
    reason: str | None = None  # failure reason (an early form of a typed failure)


class OntologyEngine:
    """Ontology engine. Holds one Ontology and offers read-only queries + binding checks."""

    def __init__(self, ontology: Ontology) -> None:
        self._onto = ontology

    @property
    def version(self) -> str:
        return self._onto.version

    # ---- entity / subtype (⪯) queries ----

    def has_entity(self, name: str) -> bool:
        return name in self._onto.entities

    def supertypes(self, name: str) -> list[str]:
        """Return all ancestor types of ``name`` along the ⪯ chain (excluding itself)."""
        result: list[str] = []
        cur = self._onto.entities[name].subtype_of if name in self._onto.entities else None
        while cur is not None:
            result.append(cur)
            cur = self._onto.entities[cur].subtype_of
        return result

    def is_subtype(self, sub: str, sup: str) -> bool:
        """Whether sub ⪯ sup holds (reflexive: X ⪯ X is true).

        e.g. is_subtype("SovereignBond", "Instrument") → True
             is_subtype("Regulator", "Instrument")     → False
        """
        if sub not in self._onto.entities or sup not in self._onto.entities:
            return False
        if sub == sup:
            return True
        return sup in self.supertypes(sub)

    def entity_types(self) -> list[str]:
        """All registered entity type names (used by grounding for candidate lookup)."""
        return list(self._onto.entities)

    def action_types(self) -> list[ActionType]:
        """All action types A (Ω's 4th primitive) — consumed by the action layer."""
        return list(self._onto.actions)

    def subtypes_of(self, name: str) -> list[str]:
        """All transitive descendants of ``name`` along ⪯ (excluding itself).

        Inverse of ``supertypes``. e.g. subtypes_of("Instrument") includes Bond, SovereignBond.
        """
        return [
            other
            for other in self._onto.entities
            if other != name and name in self.supertypes(other)
        ]

    # ---- property queries (⪯-inherited) + value validation ----

    def properties_of(self, entity_type: str) -> dict[str, Property]:
        """Effective properties: own + inherited along ⪯; a subtype overrides its supertype."""
        if entity_type not in self._onto.entities:
            return {}
        result: dict[str, Property] = {}
        # general → specific, so a subtype's redeclaration overrides the inherited one
        chain = [*reversed(self.supertypes(entity_type)), entity_type]
        for t in chain:
            for prop in self._onto.entities[t].properties:
                result[prop.name] = prop
        return result

    def property_of(self, entity_type: str, name: str) -> Property | None:
        return self.properties_of(entity_type).get(name)

    def validate_values(
        self,
        entity_type: str,
        values: Mapping[str, object],
        *,
        allow_unknown: bool = True,
    ) -> tuple[str, ...]:
        """Check ``values`` against an entity type's (inherited) properties.

        Returns typed-violation messages; an empty tuple means valid.
        """
        if entity_type not in self._onto.entities:
            return (f"unknown entity type: {entity_type}",)
        props = self.properties_of(entity_type)
        errors: list[str] = []
        for name, prop in props.items():
            if name not in values:
                if prop.required:
                    errors.append(f"missing required property: {name}")
                continue
            value = values[name]
            if not _matches_base_type(value, prop.base_type):
                got = type(value).__name__
                errors.append(f"property {name} expects {prop.base_type.value}, got {got}")
        if not allow_unknown:
            errors.extend(f"unknown property: {name}" for name in values if name not in props)
        return tuple(errors)

    # ---- verb queries ----

    def verb_class(self, verb: str) -> VerbClass | None:
        v = self._onto.verbs.get(verb)
        return v.verb_class if v is not None else None

    def verbs_of_class(self, cls: VerbClass) -> list[str]:
        """All verb names in a given act class (Vfact / Vcomm / Vinst)."""
        return [v.name for v in self._onto.verbs.values() if v.verb_class == cls]

    # ---- relation queries ----

    def relations_from(self, type_name: str) -> list[Relation]:
        """Relations whose source type is ⪯-compatible with ``type_name``."""
        return [r for r in self._onto.relations if self.is_subtype(type_name, r.from_type)]

    def relations_to(self, type_name: str) -> list[Relation]:
        """Relations whose target type is ⪯-compatible with ``type_name``."""
        return [r for r in self._onto.relations if self.is_subtype(type_name, r.to_type)]

    def relation(self, name: str) -> Relation | None:
        return next((r for r in self._onto.relations if r.name == name), None)

    def relations(self) -> list[Relation]:
        """Every relation in R (used for introspection; queries should use the ⪯-aware lookups)."""
        return list(self._onto.relations)

    def path_between(
        self, from_type: str, to_type: str, *, max_hops: int = 4, allow_narrowing: bool = False
    ) -> list[tuple[Relation, bool]] | None:
        """The shortest relation path from ``from_type`` to ``to_type``, or None if unreachable.

        Each step is ``(relation, forward)``: ``forward`` is True when the path walks the relation
        in its declared direction (from_type → to_type) and False when it walks it backwards. Both
        directions matter — "which seller sold this order" walks forward, "which orders does this
        seller appear in" walks the same relations backwards.

        This is what makes a multi-hop question answerable without anyone hand-writing the join:
        the route is derived from Ω's declared relations, so changing the ontology changes the
        route. ⪯ is respected at every step (a subtype may use its supertype's relations).

        Arriving at a *subtype* of the target always reaches it — every BulkyProduct is a Product.
        Arriving at a *supertype* does not: an Order reaches Product, but whether that product is
        a BulkyProduct is a runtime narrowing the type system cannot guarantee. Such a path is
        withheld unless ``allow_narrowing`` is set, in which case the caller takes on the check.

        Breadth-first, so the returned path is the shortest; ``max_hops`` bounds the search.
        """
        if not self.has_entity(from_type) or not self.has_entity(to_type):
            return None
        if from_type == to_type:
            return []

        def reached(landed: str) -> bool:
            if landed == to_type or self.is_subtype(landed, to_type):
                return True  # landed on the target or a subtype of it — guaranteed
            return allow_narrowing and self.is_subtype(to_type, landed)  # supertype: needs a check

        queue: list[tuple[str, list[tuple[Relation, bool]]]] = [(from_type, [])]
        seen = {from_type}
        while queue:
            current, path = queue.pop(0)
            if len(path) >= max_hops:
                continue
            steps: list[tuple[Relation, bool, str]] = [
                (r, True, r.to_type) for r in self.relations_from(current)
            ] + [(r, False, r.from_type) for r in self.relations_to(current)]
            for rel, forward, nxt in steps:
                if nxt in seen:
                    continue
                extended = [*path, (rel, forward)]
                if reached(nxt):
                    return extended
                seen.add(nxt)
                queue.append((nxt, extended))
        return None

    def needs_narrowing(self, from_type: str, to_type: str, *, max_hops: int = 4) -> bool:
        """True when the target is reachable only by narrowing to a subtype at the last step.

        Distinguishes "Ω models no route at all" from "Ω routes you to the supertype, and whether
        the value is of the subtype is a data question" — the difference between an unanswerable
        query and one that needs a runtime type check.
        """
        if self.path_between(from_type, to_type, max_hops=max_hops) is not None:
            return False
        widened = self.path_between(
            from_type, to_type, max_hops=max_hops, allow_narrowing=True
        )
        return widened is not None

    def traversable(self, from_type: str, to_type: str, *, max_hops: int = 4) -> bool:
        """Whether a path exists *and* every step declares the key it is walked by (``via``).

        A path whose relations omit ``via`` is a type-level statement only: Ω says the types are
        related but not how to follow the link in data, so it cannot be traversed.
        """
        path = self.path_between(from_type, to_type, max_hops=max_hops)
        return path is not None and all(rel.via for rel, _ in path)

    def validate_links(
        self, relation_name: str, links: Iterable[tuple[str, str]]
    ) -> tuple[str, ...]:
        """Check a set of (from_id, to_id) links against the relation's cardinality.

        Returns cardinality-violation messages; an empty tuple means valid. Identical duplicate
        links are treated as one relationship.
        """
        rel = self.relation(relation_name)
        if rel is None:
            return (f"unknown relation: {relation_name}",)
        pairs = set(links)  # dedupe: a repeated identical link is the same relationship
        errors: list[str] = []
        card = rel.effective_cardinality
        if card in (Cardinality.ONE_TO_ONE, Cardinality.MANY_TO_ONE):
            # each source may point to at most one target
            for src, n in Counter(f for f, _ in pairs).items():
                if n > 1:
                    errors.append(
                        f"{relation_name}: source '{src}' has {n} targets ({card.value})"
                    )
        if card in (Cardinality.ONE_TO_ONE, Cardinality.ONE_TO_MANY):
            # each target may be pointed to by at most one source
            for tgt, n in Counter(t for _, t in pairs).items():
                if n > 1:
                    errors.append(
                        f"{relation_name}: target '{tgt}' has {n} sources ({card.value})"
                    )
        return tuple(errors)

    # ---- typing-constraint checks (simplified CΩ) ----

    def check_binding(self, verb: str, agent_type: str, target_type: str) -> BindingCheck:
        """Check whether an agent of some type may perform a verb on a target of some type.

        This is an early form of the G1 type check: an illegal binding returns a structured
        reason, never a silent pass.
        e.g. check_binding("REGULATE", "Regulator", "Instrument") → ok=True
             check_binding("REGULATE", "Bond", "Instrument") → ok=False (Bond is not a Regulator)
        """
        if verb not in self._onto.verbs:
            return BindingCheck(ok=False, reason=f"undefined verb: {verb}")
        if not self.has_entity(agent_type):
            return BindingCheck(ok=False, reason=f"undefined agent type: {agent_type}")
        if not self.has_entity(target_type):
            return BindingCheck(ok=False, reason=f"undefined target type: {target_type}")

        applicable = [c for c in self._onto.constraints if c.verb == verb]
        if not applicable:
            # verb with no constraint: allow for now (policy tightened later)
            return BindingCheck(ok=True, rule=None)

        for c in applicable:
            agent_ok = self.is_subtype(agent_type, c.agent_must_be)
            target_ok = any(self.is_subtype(target_type, t) for t in c.target_must_be)
            if agent_ok and target_ok:
                rule = (
                    f"{verb}: agent ⪯ {c.agent_must_be} and "
                    f"target ⪯ {' ∨ '.join(c.target_must_be)}"
                )
                return BindingCheck(ok=True, rule=rule)

        # constraints exist but none satisfied → type violation (early form)
        c0 = applicable[0]
        return BindingCheck(
            ok=False,
            reason=(
                f"type_violation: {verb} requires agent ⪯ {c0.agent_must_be} and "
                f"target ⪯ {' ∨ '.join(c0.target_must_be)}; "
                f"got agent={agent_type}, target={target_type}"
            ),
        )


def _matches_base_type(value: object, base_type: BaseType) -> bool:
    """Whether a Python value conforms to a property's base type.

    ``bool`` is excluded from the numeric types (it is a subclass of ``int`` in Python), and a
    plain ``date`` is distinguished from a ``datetime`` (which is a subclass of ``date``).
    """
    if base_type is BaseType.STRING:
        return isinstance(value, str)
    if base_type is BaseType.BOOLEAN:
        return isinstance(value, bool)
    if base_type is BaseType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if base_type is BaseType.DOUBLE:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if base_type is BaseType.TIMESTAMP:
        return isinstance(value, datetime)
    if base_type is BaseType.DATE:
        return isinstance(value, date) and not isinstance(value, datetime)
    return False
