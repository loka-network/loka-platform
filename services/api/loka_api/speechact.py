"""Speech-act formalization of Workflow B.

Slide 7 does not treat a query as a flat parameter bag — it is a *speech act* q with a speaker,
a listener, a typed variable, and a predicate, dispatched to KB.DATA or KB.METHODS:

    q = asks(sp, li, ?x:T P(x))              -> P in KB.DATA:    retrieve; informs(li,sp,P(x)=v)
                                                   else               informs(li,sp, "don't know")
    q = orders(sp, li, m[in,out] P(x,m(x)))  -> m in KB.METHODS: apply;   informs(li,sp,P(x,m(x)))
                                                   else               informs(li,sp, "don't know")

    Runtime: for each informs(li,sp,P) with a concrete P  ->  add P to KB.DATA.

The runtime rule needs a qualification the usual statement leaves implicit: *which world* the
informed predicate holds in. An ``orders`` act asks what would happen under a counterfactual
dial setting, so its answer is not a fact about the actual world. KB.DATA is therefore indexed
by ``(entity, predicate, scenario)`` and every fact carries a :class:`Provenance` — ``observed``
(a reading of the world), ``derived`` (the output of a method), or ``asserted``. A derived value
can never be written into the actual world, so a projection cannot overwrite an observation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SPEAKER = "user"   # sp — the human posing the query
LISTENER = "loka"  # li — the agent answering

ACTUAL: str | None = None
"""The scenario id of the actual world. Counterfactual facts carry a non-None id."""


@dataclass(frozen=True)
class Asks:
    """asks(sp, li, ?x:T P(x)) — retrieve ``predicate`` of ``entity_id`` of type ``var_type``."""

    speaker: str
    listener: str
    var_type: str    # T — an ontology entity, e.g. "Country"
    entity_id: str   # the concrete binding of x, e.g. "ZMB"
    predicate: str   # P — an ontology attribute, e.g. "under5_mortality"

    def render(self) -> str:
        return (
            f"asks({self.speaker}, {self.listener}, "
            f"?x:{self.var_type} {self.predicate}(x={self.entity_id}))"
        )


@dataclass(frozen=True)
class Orders:
    """orders(sp, li, m[in,out] P(x, m(x))) — apply m, then report predicate P of the result."""

    speaker: str
    listener: str
    method: str                    # m — a KB.METHODS name
    in_types: tuple[str, ...]      # in
    out_type: str                  # out
    entity_id: str                 # x
    predicate: str                 # P — the outcome attribute
    args: dict[str, Any] = field(default_factory=dict)  # concrete inputs to m

    def render(self) -> str:
        sig = f"{self.method}[{','.join(self.in_types)}->{self.out_type}]"
        argstr = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return (
            f"orders({self.speaker}, {self.listener}, "
            f"{sig} {self.predicate}({self.entity_id}, {self.method}({argstr})))"
        )


@dataclass(frozen=True)
class Informs:
    """informs(li, sp, content) — the reply. ``content`` may be the string "don't know"."""

    speaker: str    # li — the informer
    listener: str   # sp — the original asker
    content: Any

    def render(self) -> str:
        c = self.content
        if isinstance(c, str):
            body = f'"{c}"' if c == "don't know" else c
        elif isinstance(c, dict) and {"entity", "predicate", "value"} <= c.keys():
            body = f"{c['predicate']}({c['entity']})={c['value']}"
        else:
            body = str(c)
        return f"informs({self.speaker}, {self.listener}, {body})"


@dataclass(frozen=True)
class Provenance:
    """Where a fact came from. ``kind`` is the load-bearing field.

    observed  — measured in the world (source + vintage identify the reading)
    derived   — produced by applying a method (method + inputs identify the run)
    asserted  — stated by a caller without further justification
    """

    kind: str
    source: str | None = None
    vintage: str | None = None
    method: str | None = None
    inputs: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "kind": self.kind,
                "source": self.source,
                "vintage": self.vintage,
                "method": self.method,
                "inputs": self.inputs,
            }.items()
            if v is not None
        }


@dataclass(frozen=True)
class Fact:
    """One entry in KB.DATA: a predicate value, its provenance, and the world it holds in."""

    entity: str
    predicate: str
    value: Any
    provenance: Provenance
    scenario_id: str | None = ACTUAL

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entity": self.entity,
            "predicate": self.predicate,
            "value": self.value,
            "provenance": self.provenance.as_dict(),
        }
        if self.scenario_id is not ACTUAL:
            d["scenario_id"] = self.scenario_id
        return d


@dataclass
class Method:
    """A KB.METHODS entry: a callable m with declared in/out types."""

    name: str
    in_types: tuple[str, ...]
    out_type: str
    fn: Callable[..., dict[str, Any]]  # returns {"value": <P(x,m(x))>, "detail": <full result>}


class KB:
    """The agent's Knowledge Base: DATA (facts) + METHODS (applicable methods).

    KB.DATA is indexed by ``(entity, predicate, scenario)``. The actual world is scenario
    ``ACTUAL``; a counterfactual produced by a method lives under its own scenario id and can
    never overwrite an observation. Retrieval defaults to the actual world.
    """

    def __init__(self) -> None:
        self.data: dict[tuple[str, str, str | None], Fact] = {}
        self.methods: dict[str, Method] = {}

    # --- KB.DATA ---
    def has_data(
        self, entity_id: str, predicate: str, scenario_id: str | None = ACTUAL
    ) -> bool:
        return (entity_id, predicate, scenario_id) in self.data

    def retrieve(
        self, entity_id: str, predicate: str, scenario_id: str | None = ACTUAL
    ) -> Any:
        fact = self.data.get((entity_id, predicate, scenario_id))
        return fact.value if fact is not None else None

    def fact(
        self, entity_id: str, predicate: str, scenario_id: str | None = ACTUAL
    ) -> Fact | None:
        return self.data.get((entity_id, predicate, scenario_id))

    def add_fact(
        self,
        entity_id: str,
        predicate: str,
        value: Any,
        provenance: Provenance | None = None,
        scenario_id: str | None = ACTUAL,
    ) -> None:
        """Runtime: for each informs(li,sp,P) -> add P to KB.DATA.

        A ``derived`` value may not be written into the actual world — that is what silently
        turned a projection into an observation. It must carry a scenario id.
        """
        prov = provenance or Provenance(kind="asserted")
        if prov.kind == "derived" and scenario_id is ACTUAL:
            raise ValueError(
                "a derived fact may not be written into the actual world; "
                "pass a scenario_id identifying the counterfactual it holds in"
            )
        self.data[(entity_id, predicate, scenario_id)] = Fact(
            entity=entity_id,
            predicate=predicate,
            value=value,
            provenance=prov,
            scenario_id=scenario_id,
        )

    def facts(
        self, scenario_id: str | None = ACTUAL, all_scenarios: bool = False
    ) -> list[dict[str, Any]]:
        """Facts in one world (the actual one by default), or every world when ``all_scenarios``."""
        return [
            f.as_dict()
            for f in self.data.values()
            if all_scenarios or f.scenario_id == scenario_id
        ]

    # --- KB.METHODS ---
    def register_method(self, m: Method) -> None:
        self.methods[m.name] = m

    def has_method(self, name: str) -> bool:
        return name in self.methods


def scenario_id_for(q: Orders) -> str:
    """A deterministic id for the counterfactual world an ``orders`` act creates."""
    payload = json.dumps({"method": q.method, "args": q.args}, sort_keys=True, default=str)
    return "cf:" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def dispatch(q: Asks | Orders, kb: KB) -> Informs:
    """Process a speech act q against the KB, returning the listener's ``informs`` reply.

    ``asks`` reads the actual world only. ``orders`` applies a method and writes the result into
    the counterfactual world it defines — never over an observation. An unsatisfiable query
    returns ``informs(li, sp, "don't know")``.
    """
    if isinstance(q, Asks):
        fact = kb.fact(q.entity_id, q.predicate)  # actual world only
        if fact is not None:
            return Informs(
                q.listener,
                q.speaker,
                {
                    "entity": q.entity_id,
                    "predicate": q.predicate,
                    "value": fact.value,
                    "provenance": fact.provenance.as_dict(),
                },
            )
        return Informs(q.listener, q.speaker, "don't know")

    # Orders
    if kb.has_method(q.method):
        result = kb.methods[q.method].fn(**q.args)
        value = result.get("value") if isinstance(result, dict) else result
        sid = scenario_id_for(q)
        kb.add_fact(
            q.entity_id,
            q.predicate,
            value,
            provenance=Provenance(kind="derived", method=q.method, inputs=dict(q.args)),
            scenario_id=sid,
        )
        return Informs(
            q.listener,
            q.speaker,
            {
                "entity": q.entity_id,
                "predicate": q.predicate,
                "value": value,
                "scenario_id": sid,
                "detail": result.get("detail") if isinstance(result, dict) else result,
            },
        )
    return Informs(q.listener, q.speaker, "don't know")
