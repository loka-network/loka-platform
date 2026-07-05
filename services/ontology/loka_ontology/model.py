"""本体论元模型(meta-schema) —— Ω = (E, V, R, ⪯, CΩ)。

这是 核心:定义"用什么数据结构表达任意一个本体论"。
注意:这里是【机器】——空的结构;具体领域的实体/关系(【内容】)由 YAML 定义文件加载,
等定了垂直领域再填。参见 内部计划文档 的"机器 vs 内容"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerbClass(str, Enum):
    """动词三分类(论文 §3.1.2 / 附录 B.2:V = Vfact ⊔ Vcomm ⊔ Vinst)。"""

    FACTUAL = "factual"  # Vfact: 作用于客观世界(TRADE, INVEST, HEDGE...)
    COMMUNICATIVE = "communicative"  # Vcomm: 社会性言语行为(ANNOUNCE, FORECAST...)
    INSTITUTIONAL = "institutional"  # Vinst: 改变规范/权限(REGULATE, VOTE, AUTHORISE...)


@dataclass(frozen=True)
class EntityType:
    """E: 一个实体类型;subtype_of 表达 ⪯(子类型偏序)。

    例:SovereignBond.subtype_of = "Bond",Bond.subtype_of = "Instrument"
    → SovereignBond ⪯ Bond ⪯ Instrument。
    """

    name: str
    subtype_of: str | None = None


@dataclass(frozen=True)
class Verb:
    """V: 一个动词及其 act class。"""

    name: str
    verb_class: VerbClass


@dataclass(frozen=True)
class Relation:
    """R: 实体类型之间的关系(有向)。例:regulator-of(Regulator → Instrument)。"""

    name: str
    from_type: str
    to_type: str


@dataclass(frozen=True)
class TypingConstraint:
    """CΩ 里的一条类型约束(简化版)。

    规定某动词的施动者类型和目标类型必须满足的子类型条件。
    例:REGULATE 要求 agent ⪯ Regulator 且 target ⪯ Instrument ∨ ⪯ PolicyLever。
    注:用简化的声明式规则起步;完整 CΩ(~250 条)后续用 Soufflé/Datalog 实现。
    """

    verb: str
    agent_must_be: str  # 施动者必须 ⪯ 这个类型
    target_must_be: tuple[str, ...]  # 目标必须 ⪯ 其中之一


@dataclass
class Ontology:
    """一个完整的本体论 Ω = (E, V, R, ⪯, CΩ)。

    ⪯(子类型偏序)编码在每个 EntityType 的 subtype_of 字段里。
    """

    version: str
    entities: dict[str, EntityType] = field(default_factory=dict)
    verbs: dict[str, Verb] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    constraints: list[TypingConstraint] = field(default_factory=list)
