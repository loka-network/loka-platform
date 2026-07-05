"""本体论加载器:把一份 YAML 定义文件加载成 Ontology 对象。

YAML 是【内容】的载体;加载器是【机器】。加载出的对象交给 OntologyEngine 使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import (
    EntityType,
    Ontology,
    Relation,
    TypingConstraint,
    Verb,
    VerbClass,
)


class OntologyLoadError(ValueError):
    """本体论定义格式错误。加载阶段就报结构化错误,不静默吞掉。"""


def load_ontology(path: str | Path) -> Ontology:
    """从 YAML 文件加载本体论。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OntologyLoadError("顶层必须是一个映射(dict)")
    return _parse(raw)


def load_ontology_str(text: str) -> Ontology:
    """从 YAML 字符串加载(便于测试)。"""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise OntologyLoadError("顶层必须是一个映射(dict)")
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> Ontology:
    version = str(raw.get("version", "v0"))

    entities: dict[str, EntityType] = {}
    for item in raw.get("entities", []) or []:
        name = item["type"]
        entities[name] = EntityType(name=name, subtype_of=item.get("subtype_of"))

    verbs: dict[str, Verb] = {}
    for item in raw.get("verbs", []) or []:
        name = item["name"]
        try:
            vclass = VerbClass(item["class"])
        except ValueError as exc:
            raise OntologyLoadError(f"动词 {name} 的 class 非法:{item.get('class')}") from exc
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


def _validate_references(onto: Ontology) -> None:
    """结构完整性检查:所有引用的类型必须存在,子类型链无环。"""
    for ent in onto.entities.values():
        if ent.subtype_of is not None and ent.subtype_of not in onto.entities:
            raise OntologyLoadError(
                f"实体 {ent.name} 的 subtype_of={ent.subtype_of} 未定义"
            )
    for rel in onto.relations:
        for t in (rel.from_type, rel.to_type):
            if t not in onto.entities:
                raise OntologyLoadError(f"关系 {rel.name} 引用了未定义的类型 {t}")
    for c in onto.constraints:
        if c.verb not in onto.verbs:
            raise OntologyLoadError(f"约束引用了未定义的动词 {c.verb}")
        for t in (c.agent_must_be, *c.target_must_be):
            if t not in onto.entities:
                raise OntologyLoadError(f"约束引用了未定义的类型 {t}")
    _check_no_cycles(onto)


def _check_no_cycles(onto: Ontology) -> None:
    """子类型链不能有环(否则 ⪯ 不是偏序)。"""
    for start in onto.entities:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise OntologyLoadError(f"子类型链存在环,涉及 {cur}")
            seen.add(cur)
            cur = onto.entities[cur].subtype_of
