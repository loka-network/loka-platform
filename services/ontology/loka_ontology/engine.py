"""本体论引擎:对加载好的 Ontology 提供查询与类型校验。

这是 对下游暴露的【公开 API】。
下游只通过这些方法访问本体论,不碰内部结构 —— 保证解耦(架构铁律#3)。

起步用简化的确定性校验;完整 CΩ(~250 条规则)后续用 Soufflé/Datalog 实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Ontology, VerbClass


@dataclass(frozen=True)
class BindingCheck:
    """一次 (verb, agent_type, target_type) 绑定校验的结果。"""

    ok: bool
    rule: str | None = None  # 命中的约束(通过时)
    reason: str | None = None  # 失败原因(typed failure 的雏形)


class OntologyEngine:
    """本体论引擎。持有一个 Ontology,提供只读查询 + 绑定校验。"""

    def __init__(self, ontology: Ontology) -> None:
        self._onto = ontology

    @property
    def version(self) -> str:
        return self._onto.version

    # ---- 实体 / 子类型查询(⪯)----

    def has_entity(self, name: str) -> bool:
        return name in self._onto.entities

    def supertypes(self, name: str) -> list[str]:
        """返回 name 的所有祖先类型(沿 ⪯ 链,不含自身)。"""
        result: list[str] = []
        cur = self._onto.entities[name].subtype_of if name in self._onto.entities else None
        while cur is not None:
            result.append(cur)
            cur = self._onto.entities[cur].subtype_of
        return result

    def is_subtype(self, sub: str, sup: str) -> bool:
        """sub ⪯ sup 是否成立(含自反:X ⪯ X 为真)。

        例:is_subtype("SovereignBond", "Instrument") → True
            is_subtype("Regulator", "Instrument")     → False
        """
        if sub not in self._onto.entities or sup not in self._onto.entities:
            return False
        if sub == sup:
            return True
        return sup in self.supertypes(sub)

    # ---- 动词查询 ----

    def verb_class(self, verb: str) -> VerbClass | None:
        v = self._onto.verbs.get(verb)
        return v.verb_class if v is not None else None

    # ---- 类型约束校验(CΩ 简化版)----

    def check_binding(self, verb: str, agent_type: str, target_type: str) -> BindingCheck:
        """校验"某类型的施动者对某类型的目标执行某动词"是否合法。

        这是 G1 类型检查的雏形:非法绑定返回结构化 reason,不静默通过。
        例:check_binding("REGULATE", "Regulator", "Instrument") → ok=True
            check_binding("REGULATE", "Bond", "Instrument")      → ok=False(Bond 不是 Regulator)
        """
        if verb not in self._onto.verbs:
            return BindingCheck(ok=False, reason=f"未定义的动词:{verb}")
        if not self.has_entity(agent_type):
            return BindingCheck(ok=False, reason=f"未定义的施动者类型:{agent_type}")
        if not self.has_entity(target_type):
            return BindingCheck(ok=False, reason=f"未定义的目标类型:{target_type}")

        applicable = [c for c in self._onto.constraints if c.verb == verb]
        if not applicable:
            # 无约束的动词:起步先放行(完整策略后续收紧)
            return BindingCheck(ok=True, rule=None)

        for c in applicable:
            agent_ok = self.is_subtype(agent_type, c.agent_must_be)
            target_ok = any(self.is_subtype(target_type, t) for t in c.target_must_be)
            if agent_ok and target_ok:
                rule = (
                    f"{verb}: agent ⪯ {c.agent_must_be} 且 "
                    f"target ⪯ {' ∨ '.join(c.target_must_be)}"
                )
                return BindingCheck(ok=True, rule=rule)

        # 有约束但都不满足 → 类型违规(type_violation 雏形)
        c0 = applicable[0]
        return BindingCheck(
            ok=False,
            reason=(
                f"type_violation: {verb} 要求 agent ⪯ {c0.agent_must_be} 且 "
                f"target ⪯ {' ∨ '.join(c0.target_must_be)};"
                f"实际 agent={agent_type}, target={target_type}"
            ),
        )
