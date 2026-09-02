"""The three-role contract, served as decisions rather than as prose.

roles.md is a governance specification: who may say what, under which condition, and what has to
be reported alongside what. Two of its sentences decide the shape of this module.

    判定由代码依据冻结口径完成，你的作用是解释与取舍。
    处置（必须落在代码算出的可行集合内，越界则拒收）

So what is served here is not a conclusion. It is the boundary a conclusion has to sit inside:
whether this role may perform this act at all, whether what it hands on carries every field the
contract names, and what the target number is. The role explains and chooses; this decides
admissibility.

What is deliberately absent: the map from measurements to the four dispositions. roles.md says
that map is frozen by a person and never derived, and inventing one here would be the thing it
names as the entrance to every later evasion.
"""

from __future__ import annotations

import os
import re
from typing import Any

#: The one derivation roles.md states exactly: close a third of the distance to 100.
#: Written as a closed fraction rather than "improve by N points", because a number of points
#: means something different at every baseline and drifts as the baseline moves.
def target_number(baseline: float) -> dict[str, Any]:
    """X → X + (100 − X)/3, with the arithmetic shown rather than asserted."""
    if not 0.0 <= baseline <= 100.0:
        raise ValueError(f"baseline must be a percentage in [0, 100]; got {baseline}")
    gap = 100.0 - baseline
    return {
        "baseline": baseline,
        "gap_to_100": round(gap, 4),
        "closes": "one third of the gap",
        "target": round(baseline + gap / 3.0, 4),
        "formula": "X + (100 - X) / 3",
    }


_ROLE_OUTPUT = {
    "Biologist": "BiologistOutput",
    "Chemist": "ChemistOutput",
    "AIExpert": "AIExpertOutput",
}


def output_entity_for(role: str) -> str | None:
    """The entity type whose required attributes are that role's handoff contract."""
    return _ROLE_OUTPUT.get(role)


#: Registry columns that identify a paper. A citation resolves when it names one of these.
_ID_COLUMNS = ("paper_id", "doi", "arxiv_id", "pmcid")


def load_registry(root: str | None = None) -> dict[str, dict[str, str]]:
    """Every paper the corpus holds, keyed by each identifier that resolves to it.

    A citation is checked against this rather than trusted. roles.md says an unresolvable one is
    dropped whole — which is only enforceable if there is something to resolve against, and is
    otherwise a promise about care.
    """
    root = root or os.getenv("LOKA_PAPERS", "")
    if not root:
        return {}
    path = os.path.join(root, "_index", "registry.tsv")
    if not os.path.exists(path):
        return {}

    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(header, values))
            for column in _ID_COLUMNS:
                key = (row.get(column) or "").strip()
                if key:
                    out[key.lower()] = row
    return out


def resolve_citations(citations: list[str], registry: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Split citations into those the corpus holds and those it does not.

    The unresolved are named, not counted. A count says how many were dropped; the names say
    which, and only the second lets anyone check whether the drop was right.
    """
    resolved, unresolved = [], []
    for c in citations:
        row = registry.get(str(c).strip().lower())
        if row is None:
            unresolved.append(c)
        else:
            resolved.append({"cited_as": c, "paper_id": row.get("paper_id"),
                             "title": row.get("title"), "roles": row.get("roles")})
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "registry_size": len(registry),
    }


_PREDICATE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")
_OPS = {
    ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def evaluate_falsifier(condition: str, measurements: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a falsification predicate against measured values.

    This is the step that keeps the arrangement from being a report generator: the condition is
    consumed after the numbers exist, and the verdict is recorded. A condition that cannot be
    read is reported as unevaluable — never as satisfied, and never as refuted, because both
    would be this code deciding something it cannot see.
    """
    m = _PREDICATE.match(condition or "")
    if not m:
        return {
            "condition": condition,
            "evaluable": False,
            "reason": (
                "not of a readable shape; expected '<name> <op> <number>' with op in "
                f"{sorted(_OPS)}"
            ),
        }
    name, op, threshold = m.group(1), m.group(2), float(m.group(3))
    if name not in measurements:
        return {
            "condition": condition,
            "evaluable": False,
            "reason": f"no measurement named {name!r} was supplied",
            "supplied": sorted(measurements),
        }
    try:
        observed = float(measurements[name])
    except (TypeError, ValueError):
        return {"condition": condition, "evaluable": False,
                "reason": f"{name}={measurements[name]!r} is not a number"}

    return {
        "condition": condition,
        "evaluable": True,
        "observed": {name: observed},
        "threshold": threshold,
        "falsified": bool(_OPS[op](observed, threshold)),
    }
