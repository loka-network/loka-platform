"""Ontology loader: parse a YAML definition file into an Ontology object.

The YAML file carries the content; the loader is the machinery. The loaded object is
consumed by the OntologyEngine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import (
    BaseType,
    EntityType,
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

    relations: list[Relation] = [
        Relation(name=item["name"], from_type=item["from"], to_type=item["to"])
        for item in raw.get("relations", []) or []
    ]

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

    onto = Ontology(
        version=version,
        entities=entities,
        verbs=verbs,
        relations=relations,
        constraints=constraints,
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
    _check_no_cycles(onto)


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
