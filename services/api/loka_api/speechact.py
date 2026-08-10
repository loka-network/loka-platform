"""Speech-act formalization of Workflow B (Sifakis, slide 7 "Queries").

Slide 7 does not treat a query as a flat parameter bag — it is a *speech act* q with a speaker,
a listener, a typed variable, and a predicate, dispatched to KB.DATA or KB.METHODS:

    q = asks(sp, li, ?x:T P(x))                -> if P in KB.DATA:    retrieve; informs(li,sp, P(x)=v)
                                                   else               informs(li,sp, "don't know")
    q = orders(sp, li, m[in,out] P(x, m(x)))   -> if m in KB.METHODS: apply m; informs(li,sp, P(x,m(x)))
                                                   else               informs(li,sp, "don't know")

    Runtime: for each informs(li,sp,P) with a concrete P  ->  add P to KB.DATA.

This is load-bearing, not decoration:
  * the variable is typed against the ontology (``?x:Country``);
  * the predicate P must be an ontology attribute;
  * the method m must be registered in KB.METHODS.
A query whose predicate/method the KB cannot satisfy is refused with ``informs(li,sp,"don't know")``
— the agent's honest limit, exactly as the professor wrote it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SPEAKER = "user"   # sp — the human posing the query
LISTENER = "loka"  # li — the agent answering


@dataclass(frozen=True)
class Asks:
    """asks(sp, li, ?x:T P(x)) — retrieve DATA attribute ``predicate`` of entity ``entity_id``:``var_type``."""

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
    """orders(sp, li, m[in,out] P(x, m(x))) — apply method m, then report predicate P of the result."""

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
    """informs(li, sp, content) — the listener's reply. ``content`` may be the string "don't know"."""

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


@dataclass
class Method:
    """A KB.METHODS entry: a callable m with declared in/out types."""

    name: str
    in_types: tuple[str, ...]
    out_type: str
    fn: Callable[..., dict[str, Any]]  # returns {"value": <P(x,m(x))>, "detail": <full result>}


class KB:
    """The agent's Knowledge Base: DATA (facts) + METHODS (applicable methods) — slide 7's KB."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], Any] = {}   # (entity_id, predicate) -> value
        self.methods: dict[str, Method] = {}

    # --- KB.DATA ---
    def has_data(self, entity_id: str, predicate: str) -> bool:
        return (entity_id, predicate) in self.data

    def retrieve(self, entity_id: str, predicate: str) -> Any:
        return self.data.get((entity_id, predicate))

    def add_fact(self, entity_id: str, predicate: str, value: Any) -> None:
        """Runtime: for each informs(li,sp,P) -> add P to KB.DATA."""
        self.data[(entity_id, predicate)] = value

    def facts(self) -> list[dict[str, Any]]:
        return [
            {"entity": e, "predicate": p, "value": v} for (e, p), v in self.data.items()
        ]

    # --- KB.METHODS ---
    def register_method(self, m: Method) -> None:
        self.methods[m.name] = m

    def has_method(self, name: str) -> bool:
        return name in self.methods


def dispatch(q: Asks | Orders, kb: KB) -> Informs:
    """Process a speech act q against the KB, returning the listener's ``informs`` reply.

    Every concrete answer is written back into KB.DATA (the professor's runtime rule), so the KB
    grows as it is queried. An unsatisfiable query returns ``informs(li, sp, "don't know")``.
    """
    if isinstance(q, Asks):
        if kb.has_data(q.entity_id, q.predicate):
            value = kb.retrieve(q.entity_id, q.predicate)
            kb.add_fact(q.entity_id, q.predicate, value)  # informs -> add P to KB (idempotent here)
            return Informs(q.listener, q.speaker,
                           {"entity": q.entity_id, "predicate": q.predicate, "value": value})
        return Informs(q.listener, q.speaker, "don't know")

    # Orders
    if kb.has_method(q.method):
        result = kb.methods[q.method].fn(**q.args)
        value = result.get("value") if isinstance(result, dict) else result
        kb.add_fact(q.entity_id, q.predicate, value)  # runtime: add P to KB
        return Informs(q.listener, q.speaker, {
            "entity": q.entity_id, "predicate": q.predicate, "value": value,
            "detail": result.get("detail") if isinstance(result, dict) else result,
        })
    return Informs(q.listener, q.speaker, "don't know")
