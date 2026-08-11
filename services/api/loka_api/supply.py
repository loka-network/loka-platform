"""Supply scenario — eligibility from a guard, consequence along the relations.

The health scenario exercises one entity and no relations, so Ω's R and ⪯ never carry weight
there. This one asks a question a single table cannot answer: *tighten the rule for what may ship
on the standard service, and tell me what it touches.* Answering it uses four parts of Ω at once —

  Actions   the rule being changed is an action's ``guard``, declared in Ω, not a constant here
  A         the guard names an attribute, which must be declared on the target entity
  ⪯         a subtype is included when its supertype is read (a BulkyProduct is a Product)
  R         the consequence is followed along the declared relations, by their declared ``via``

so no join and no eligibility rule is written in application code: change the ontology and the
answer changes with it.
"""

from __future__ import annotations

import csv
import operator
import os
import re
from collections.abc import Callable
from typing import Any

_GUARD_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")

# The sample dataset used when no CSVs are present, so the scenario is demonstrable offline.
# Rows are keyed by entity type — the shape the adapters already use.
_SAMPLE: dict[str, list[dict[str, Any]]] = {
    "Seller": [
        {"seller_id": "s_A", "seller_state": "SP", "on_time_rate": 0.91},
        {"seller_id": "s_B", "seller_state": "RJ", "on_time_rate": 0.74},
    ],
    "Product": [
        {"product_id": "p_1", "seller_id": "s_A", "weight_g": 1200.0, "category": "electronics"},
        {"product_id": "p_2", "seller_id": "s_B", "weight_g": 8000.0, "category": "furniture"},
    ],
    "BulkyProduct": [
        {"product_id": "p_3", "seller_id": "s_A", "weight_g": 45000.0, "category": "appliance"},
        {"product_id": "p_4", "seller_id": "s_B", "weight_g": 32000.0, "category": "furniture"},
    ],
    "Order": [
        {"order_id": "o_1", "product_id": "p_1", "customer_id": "c_X", "days_late": -2.0},
        {"order_id": "o_2", "product_id": "p_3", "customer_id": "c_Y", "days_late": 6.0},
        {"order_id": "o_3", "product_id": "p_4", "customer_id": "c_Z", "days_late": 11.0},
        {"order_id": "o_4", "product_id": "p_2", "customer_id": "c_X", "days_late": 1.0},
    ],
    "Customer": [
        {"customer_id": "c_X", "customer_state": "RJ"},
        {"customer_id": "c_Y", "customer_state": "SP"},
        {"customer_id": "c_Z", "customer_state": "MG"},
    ],
}


def load_supply_ontology() -> Any | None:
    """Load supply-v1 (env override, else the repo's examples/)."""
    from loka_ontology import OntologyEngine, load_ontology_str

    here = os.path.dirname(__file__)
    for p in (
        os.getenv("LOKA_SUPPLY_ONTOLOGY"),
        os.path.join(here, "..", "..", "..", "examples", "supply_ontology.yaml"),
        os.path.join(os.getcwd(), "examples", "supply_ontology.yaml"),
    ):
        if p and os.path.exists(p):
            with open(p) as f:
                return OntologyEngine(load_ontology_str(f.read()))
    return None


def load_supply_dataset(engine: Any | None = None) -> dict[str, list[dict[str, Any]]]:
    """Rows per entity type.

    Reads ``<dir>/<EntityType>.csv`` from ``LOKA_SUPPLY_DATA`` when set, so a real dataset drops
    in without code changes; otherwise returns the built-in sample. Numeric strings are converted
    so guards and comparisons operate on numbers, not text.
    """
    directory = os.getenv("LOKA_SUPPLY_DATA")
    if not directory or not os.path.isdir(directory):
        return {k: [dict(r) for r in v] for k, v in _SAMPLE.items()}

    types = engine.entity_types() if engine is not None else list(_SAMPLE)
    data: dict[str, list[dict[str, Any]]] = {}
    for entity in types:
        path = os.path.join(directory, f"{entity}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data[entity] = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return data


def _coerce(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def parse_guard(guard: str) -> tuple[str, str, float] | None:
    """Split a numeric guard into (attribute, op, threshold), else None."""
    m = _GUARD_RE.match(guard or "")
    if not m:
        return None
    return m.group(1), m.group(2), float(m.group(3))


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}


def impact_of_tightening(
    engine: Any,
    dataset: dict[str, list[dict[str, Any]]],
    *,
    action_name: str,
    new_threshold: float,
    propagate_to: list[str] | None = None,
) -> dict[str, Any]:
    """Which entities lose eligibility under a tighter guard, and what that reaches.

    The rule is Ω's: the action, its target type and its guard all come from the ontology, and the
    guard's attribute must be declared on the target. Rows that satisfied the old threshold but
    not the new one are the ones that lose eligibility; the consequence is then followed along the
    declared relations to each requested type, reporting the route as well as the rows.
    """
    from loka_ontology.traverse import reach, rows_of_type

    action = next((a for a in engine.action_types() if a.name == action_name), None)
    if action is None:
        known = sorted(a.name for a in engine.action_types())
        return {"error": f"'{action_name}' is not an action in ontology {engine.version}",
                "known_actions": known}

    parsed = parse_guard(action.guard)
    if parsed is None:
        return {"error": f"action {action_name} has no numeric guard to tighten",
                "guard": action.guard}
    attribute, op, old_threshold = parsed

    if attribute not in engine.properties_of(action.target):
        return {"error": f"guard references '{attribute}', which {action.target} does not declare "
                         f"in ontology {engine.version}"}

    compare = _OPS[op]
    candidates = rows_of_type(engine, dataset, action.target)
    newly_ineligible = [
        r for r in candidates
        if isinstance(r.get(attribute), (int, float))
        and compare(float(r[attribute]), old_threshold)      # was allowed
        and not compare(float(r[attribute]), new_threshold)  # no longer is
    ]

    targets = propagate_to or [t for t in engine.entity_types() if t != action.target]
    consequences = []
    for target in targets:
        out = reach(engine, dataset, from_type=action.target, to_type=target,
                    start=newly_ineligible)
        if out.get("route") is None:
            continue
        consequences.append({
            "entity": target,
            "route": out["route"],
            "hops": out["hops"],
            "requires_narrowing": out.get("requires_narrowing", False),
            "affected": out["rows"],
            "count": len(out["rows"]),
        })

    return {
        "ontology_version": engine.version,
        "action": action_name,
        "target_entity": action.target,
        "guard": {"attribute": attribute, "operator": op,
                  "from": old_threshold, "to": new_threshold},
        "newly_ineligible": newly_ineligible,
        "newly_ineligible_count": len(newly_ineligible),
        "consequences": consequences,
    }
