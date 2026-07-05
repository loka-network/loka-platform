"""验收测试:本体论引擎骨架能跑。

演示目标(内部计划文档):加载 toy 本体论 → 回答子类型问题 → 校验类型绑定。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loka_ontology import OntologyEngine, load_ontology
from loka_ontology.loader import OntologyLoadError, load_ontology_str

TOY = Path(__file__).parent.parent / "examples" / "toy_ontology.yaml"


@pytest.fixture
def engine() -> OntologyEngine:
    return OntologyEngine(load_ontology(TOY))


# ---- 子类型 ⪯ ----

def test_subtype_chain_true(engine: OntologyEngine) -> None:
    # SovereignBond ⪯ Bond ⪯ Instrument
    assert engine.is_subtype("SovereignBond", "Instrument") is True
    assert engine.is_subtype("SovereignBond", "Bond") is True


def test_subtype_reflexive(engine: OntologyEngine) -> None:
    assert engine.is_subtype("Bond", "Bond") is True


def test_subtype_false(engine: OntologyEngine) -> None:
    assert engine.is_subtype("Regulator", "Instrument") is False
    assert engine.is_subtype("Instrument", "SovereignBond") is False  # 方向相反


def test_supertypes(engine: OntologyEngine) -> None:
    assert engine.supertypes("SovereignBond") == ["Bond", "Instrument"]


# ---- 动词 ----

def test_verb_class(engine: OntologyEngine) -> None:
    from loka_ontology import VerbClass

    assert engine.verb_class("REGULATE") == VerbClass.INSTITUTIONAL
    assert engine.verb_class("TRADE") == VerbClass.FACTUAL
    assert engine.verb_class("NONEXISTENT") is None


# ---- 类型约束校验(CΩ 雏形)----

def test_binding_ok(engine: OntologyEngine) -> None:
    # CentralBank ⪯ Regulator,SovereignBond ⪯ Instrument → 合法
    res = engine.check_binding("REGULATE", "CentralBank", "SovereignBond")
    assert res.ok is True
    assert res.rule is not None


def test_binding_agent_violation(engine: OntologyEngine) -> None:
    # Bond 不是 Regulator → 非法(type_violation)
    res = engine.check_binding("REGULATE", "Bond", "Instrument")
    assert res.ok is False
    assert res.reason is not None and "type_violation" in res.reason


def test_binding_unknown_verb(engine: OntologyEngine) -> None:
    res = engine.check_binding("FLY", "Regulator", "Instrument")
    assert res.ok is False
    assert res.reason is not None and "未定义的动词" in res.reason


# ---- 加载器结构校验 ----

def test_loader_rejects_dangling_subtype() -> None:
    bad = "version: v0\nentities:\n  - {type: Bond, subtype_of: DoesNotExist}\n"
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)


def test_loader_rejects_cycle() -> None:
    bad = (
        "version: v0\nentities:\n"
        "  - {type: A, subtype_of: B}\n"
        "  - {type: B, subtype_of: A}\n"
    )
    with pytest.raises(OntologyLoadError):
        load_ontology_str(bad)
