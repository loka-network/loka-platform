"""Ontology engine: queries and type checks over a loaded Ontology.

This is the public API the engine exposes to downstream services (grounding, compiler).
Downstream accesses the ontology only through these methods, never its internals — this is
what keeps the system decoupled (engineering principle #3).

The checks start as a simplified deterministic form; the full CΩ (~250 rules) will be
implemented with Soufflé/Datalog later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Ontology, VerbClass


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

    # ---- verb queries ----

    def verb_class(self, verb: str) -> VerbClass | None:
        v = self._onto.verbs.get(verb)
        return v.verb_class if v is not None else None

    # ---- typing-constraint checks (simplified CΩ) ----

    def check_binding(self, verb: str, agent_type: str, target_type: str) -> BindingCheck:
        """Check whether an agent of some type may perform a verb on a target of some type.

        This is an early form of the G1 type check: an illegal binding returns a structured
        reason, never a silent pass.
        e.g. check_binding("REGULATE", "Regulator", "Instrument") → ok=True
             check_binding("REGULATE", "Bond", "Instrument")      → ok=False (Bond is not a Regulator)
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
