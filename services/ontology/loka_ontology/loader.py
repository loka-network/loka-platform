"""Ontology loader: parse a YAML definition file into an Ontology object.

The YAML file carries the content; the loader is the machinery. The loaded object is
consumed by the OntologyEngine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import (
    ActionType,
    BaseType,
    Cardinality,
    EntityType,
    Norm,
    NormStatus,
    Ontology,
    Property,
    Relation,
    TypingConstraint,
    Verb,
    VerbClass,
)


class OntologyLoadError(ValueError):
    """Malformed ontology definition. Fail with a structured error at load time; never swallow."""


def load_ontology(path: str | Path) -> Ontology:
    """Load an ontology from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OntologyLoadError("top level must be a mapping (dict)")
    return _parse(raw)


def load_ontology_str(text: str) -> Ontology:
    """Load an ontology from a YAML string (convenient for tests)."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise OntologyLoadError("top level must be a mapping (dict)")
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> Ontology:
    version = str(raw.get("version", "v0"))

    entities: dict[str, EntityType] = {}
    for item in raw.get("entities", []) or []:
        name = item["type"]
        entities[name] = EntityType(
            name=name,
            subtype_of=item.get("subtype_of"),
            properties=_parse_properties(name, item.get("properties", []) or []),
            backing=item.get("backing"),
        )

    verbs: dict[str, Verb] = {}
    for item in raw.get("verbs", []) or []:
        name = item["name"]
        try:
            vclass = VerbClass(item["class"])
        except ValueError as exc:
            raise OntologyLoadError(
                f"verb {name} has an invalid class: {item.get('class')}"
            ) from exc
        verbs[name] = Verb(name=name, verb_class=vclass)

    relations: list[Relation] = []
    for item in raw.get("relations", []) or []:
        raw_card = item.get("cardinality")
        try:
            # Absent stays absent: "the ontology did not say" is distinct from "the ontology said
            # unconstrained", and review needs to tell them apart. Enforcement reads
            # ``effective_cardinality``, which resolves the absent case to many-to-many.
            cardinality = Cardinality(raw_card) if raw_card is not None else None
        except ValueError as exc:
            raise OntologyLoadError(
                f"relation {item['name']} has invalid cardinality: {raw_card}"
            ) from exc
        relations.append(
            Relation(
                name=item["name"],
                from_type=item["from"],
                to_type=item["to"],
                cardinality=cardinality,
                via=item.get("via"),
            )
        )

    constraints: list[TypingConstraint] = []
    for item in raw.get("constraints", []) or []:
        target = item["target_must_be"]
        target_tuple = tuple(target) if isinstance(target, list) else (target,)
        constraints.append(
            TypingConstraint(
                verb=item["verb"],
                agent_must_be=item["agent_must_be"],
                target_must_be=target_tuple,
            )
        )

    actions: list[ActionType] = []
    for item in raw.get("actions", []) or []:
        actions.append(
            ActionType(
                name=item["name"],
                verb=item["verb"],
                target=item["target"],
                guard=item.get("guard", ""),
                effect=item.get("effect", ""),
                controllable=bool(item.get("controllable", True)),
            )
        )

    norms: list[Norm] = []
    for item in raw.get("norms", []) or []:
        raw_status = str(item["status"])
        try:
            status = NormStatus(raw_status)
        except ValueError:
            allowed = ", ".join(sorted(s.value for s in NormStatus))
            raise OntologyLoadError(
                f"norm {item.get('name', '?')} has status {raw_status!r}; expected one of {allowed}"
            ) from None
        norms.append(
            Norm(
                name=str(item.get("name") or f"{item['action']}:{raw_status}"),
                action=str(item["action"]),
                status=status,
                when=str(item.get("when", "")),
                rationale=str(item.get("rationale", "")),
            )
        )

    onto = Ontology(
        version=version,
        entities=entities,
        verbs=verbs,
        relations=relations,
        constraints=constraints,
        actions=actions,
        norms=norms,
    )
    _validate_references(onto)
    return onto


def _parse_properties(entity: str, items: list[dict[str, Any]]) -> tuple[Property, ...]:
    props: list[Property] = []
    seen: set[str] = set()
    for p in items:
        pname = p["name"]
        if pname in seen:
            raise OntologyLoadError(f"entity {entity} has duplicate property {pname}")
        seen.add(pname)
        try:
            base_type = BaseType(p["type"])
        except ValueError as exc:
            raise OntologyLoadError(
                f"entity {entity} property {pname} has invalid type: {p.get('type')}"
            ) from exc
        props.append(
            Property(
                name=pname,
                base_type=base_type,
                required=bool(p.get("required", False)),
                description=p.get("description"),
            )
        )
    return tuple(props)


def _validate_references(onto: Ontology) -> None:
    """Structural integrity: every referenced type must exist; subtype chains must be acyclic."""
    for ent in onto.entities.values():
        if ent.subtype_of is not None and ent.subtype_of not in onto.entities:
            raise OntologyLoadError(
                f"entity {ent.name} has subtype_of={ent.subtype_of}, which is not defined"
            )
    for rel in onto.relations:
        for t in (rel.from_type, rel.to_type):
            if t not in onto.entities:
                raise OntologyLoadError(f"relation {rel.name} references undefined type {t}")
    for c in onto.constraints:
        if c.verb not in onto.verbs:
            raise OntologyLoadError(f"constraint references undefined verb {c.verb}")
        for t in (c.agent_must_be, *c.target_must_be):
            if t not in onto.entities:
                raise OntologyLoadError(f"constraint references undefined type {t}")
    for a in onto.actions:
        if a.verb not in onto.verbs:
            raise OntologyLoadError(f"action {a.name} references undefined verb {a.verb}")
        if a.target not in onto.entities:
            raise OntologyLoadError(f"action {a.name} references undefined target {a.target}")
    _check_no_cycles(onto)
    _check_override_compatibility(onto)
    _check_relation_keys(onto)
    _check_norms(onto)


def _check_norms(onto: Ontology) -> None:
    """A norm must govern a declared, controllable action (CΩ R9, R10).

    R9 — an undeclared action. A norm naming an action Ω does not define is a rule about nothing:
    it can never fire, so it silently grants exactly the permission it was written to withhold.
    Catching it at load time is the difference between a governed system and one that believes
    it is governed.

    R10 — an uncontrollable action. N(s, ac) is defined on the controllable half of A. Obliging
    or forbidding something nobody chose has no addressee; whoever wrote it meant something else,
    and should be told rather than have it accepted and ignored.
    """
    by_name = {a.name: a for a in onto.actions}
    for n in onto.norms:
        action = by_name.get(n.action)
        if action is None:
            raise OntologyLoadError(
                f"norm {n.name} governs action {n.action}, which is not defined"
            )
        if not action.controllable:
            raise OntologyLoadError(
                f"norm {n.name} governs {n.action}, which is uncontrollable (in Au); "
                "norms are defined on controllable actions"
            )


def _check_relation_keys(onto: Ontology) -> None:
    """A relation's ``via`` field must be declared on both types it connects (CΩ R8).

    ``via`` names the field the link is carried by, and a link is followed by matching that field
    on both sides. If either side does not declare it, a path through this relation is computable
    but not followable — the ontology would promise a route it cannot walk. Checking it at load
    time means a traversal derived from Ω is guaranteed to have the fields it needs.
    """
    def effective(entity: str) -> set[str]:
        names: set[str] = set()
        cur: str | None = entity
        while cur is not None:
            names.update(p.name for p in onto.entities[cur].properties)
            cur = onto.entities[cur].subtype_of
        return names

    for rel in onto.relations:
        if rel.via is None:
            continue  # a type-level relation; declared as not traversable
        for side in (rel.from_type, rel.to_type):
            if rel.via not in effective(side):
                raise OntologyLoadError(
                    f"relation {rel.name} is traversed via '{rel.via}', but {side} does not "
                    f"declare that property; a link field must exist on both types it connects"
                )


def _check_no_cycles(onto: Ontology) -> None:
    """Subtype chains must not contain cycles (otherwise ⪯ is not a partial order)."""
    for start in onto.entities:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise OntologyLoadError(f"subtype chain contains a cycle involving {cur}")
            seen.add(cur)
            cur = onto.entities[cur].subtype_of


# Which base types a subtype may narrow an inherited property to. Identity is always allowed;
# these are the additional widening→narrowing pairs that preserve substitutability (every
# INTEGER is a valid DOUBLE; every DATE is a valid TIMESTAMP), so a value of the subtype's
# type is still a legal value of the supertype's type.
_NARROWABLE_TO: dict[BaseType, frozenset[BaseType]] = {
    BaseType.DOUBLE: frozenset({BaseType.INTEGER}),
    BaseType.TIMESTAMP: frozenset({BaseType.DATE}),
}


def _check_override_compatibility(onto: Ontology) -> None:
    """A subtype redeclaring an inherited property must not break substitutability (⪯ soundness).

    If ``Sub ⪯ Super`` and both declare property ``p``, then a ``Sub`` must remain usable wherever
    a ``Super`` is expected. That fails if the override changes ``p`` to an unrelated type (e.g.
    ``double`` -> ``string``), or relaxes a required property to optional. Widening the type
    (``integer`` -> ``double``) also fails: the subtype would admit values the supertype forbids.
    Only an identical type or a sound narrowing is accepted.
    """
    for ent in onto.entities.values():
        own = {p.name: p for p in ent.properties}
        if not own:
            continue
        ancestor = ent.subtype_of
        while ancestor is not None:
            parent = onto.entities[ancestor]
            for p in parent.properties:
                sub_prop = own.get(p.name)
                if sub_prop is None:
                    continue
                allowed = _NARROWABLE_TO.get(p.base_type, frozenset())
                if sub_prop.base_type != p.base_type and sub_prop.base_type not in allowed:
                    raise OntologyLoadError(
                        f"entity {ent.name} overrides inherited property {p.name} with type "
                        f"{sub_prop.base_type} but {parent.name} declares it as {p.base_type}; "
                        f"a subtype may only repeat the type or narrow it "
                        f"(substitutability of {ent.name} for {parent.name} would break)"
                    )
                if p.required and not sub_prop.required:
                    raise OntologyLoadError(
                        f"entity {ent.name} overrides inherited property {p.name} as optional but "
                        f"{parent.name} declares it required; a subtype may not relax a "
                        f"requirement of its supertype"
                    )
            ancestor = parent.subtype_of
