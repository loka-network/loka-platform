"""Soufflé-backed CΩ type checker — the full-strength ontology type check.

Compiles the ontology's typing constraints (CΩ) into a Datalog program and evaluates it with
Soufflé (the whitepaper's choice: decidable, terminating, ~100x faster than relational joins
at scale). The subtype relation ⪯ is the transitive closure computed in Datalog.

The engine's pure-Python ``check_binding`` remains the dependency-free default; this checker is
used where the ``souffle`` binary is available and produces identical verdicts.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .engine import BindingCheck
from .model import Ontology

Query = tuple[str, str, str]  # (verb, agent_type, target_type)

_PROGRAM = """\
.decl type(t:symbol)
.decl subtype_direct(sub:symbol, sup:symbol)
.decl verb(v:symbol)
.decl constraint(v:symbol, agent_type:symbol, target_type:symbol)
.decl query(v:symbol, a:symbol, t:symbol)
.input type
.input subtype_direct
.input verb
.input constraint
.input query

// subtype ⪯ : reflexive-transitive closure of subtype_direct
.decl subtype(sub:symbol, sup:symbol)
subtype(x, x) :- type(x).
subtype(x, y) :- subtype_direct(x, y).
subtype(x, z) :- subtype_direct(x, y), subtype(y, z).

.decl has_constraint(v:symbol)
has_constraint(v) :- constraint(v, _, _).

.decl satisfied(v:symbol, a:symbol, t:symbol)
satisfied(v, a, t) :- query(v, a, t), constraint(v, ca, ct), subtype(a, ca), subtype(t, ct).

.decl admissible(v:symbol, a:symbol, t:symbol)
admissible(v, a, t) :- satisfied(v, a, t).
admissible(v, a, t) :- query(v, a, t), verb(v), !has_constraint(v).

.decl violation(v:symbol, a:symbol, t:symbol)
violation(v, a, t) :- query(v, a, t), verb(v), !admissible(v, a, t).

.decl unknown_verb(v:symbol, a:symbol, t:symbol)
unknown_verb(v, a, t) :- query(v, a, t), !verb(v).

.decl mentioned(x:symbol)
mentioned(a) :- query(_, a, _).
mentioned(t) :- query(_, _, t).
.decl unknown_type(x:symbol)
unknown_type(x) :- mentioned(x), !type(x).

.output admissible
.output violation
.output unknown_verb
.output unknown_type
"""


class SouffleUnavailable(RuntimeError):
    """The ``souffle`` binary is not available on PATH."""


def souffle_available() -> bool:
    return shutil.which("souffle") is not None


class SouffleTypeChecker:
    """Evaluates CΩ binding admissibility with Soufflé."""

    def __init__(self, ontology: Ontology) -> None:
        if not souffle_available():
            raise SouffleUnavailable("the 'souffle' binary is not on PATH")
        self._onto = ontology

    def check_binding(self, verb: str, agent_type: str, target_type: str) -> BindingCheck:
        return self.check_bindings([(verb, agent_type, target_type)])[
            (verb, agent_type, target_type)
        ]

    def check_bindings(self, queries: Iterable[Query]) -> dict[Query, BindingCheck]:
        qlist = list(queries)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fdir, odir = root / "facts", root / "out"
            fdir.mkdir()
            odir.mkdir()
            self._write_facts(fdir, qlist)
            (root / "comega.dl").write_text(_PROGRAM, encoding="utf-8")
            subprocess.run(
                ["souffle", "-F", str(fdir), "-D", str(odir), str(root / "comega.dl")],
                check=True,
                capture_output=True,
                text=True,
            )
            admissible = _read_tuples(odir / "admissible.csv")
            violation = _read_tuples(odir / "violation.csv")
            unknown_verb = _read_tuples(odir / "unknown_verb.csv")
            unknown_type = {row[0] for row in _read_tuples(odir / "unknown_type.csv")}

        results: dict[Query, BindingCheck] = {}
        for q in qlist:
            verb, agent, target = q
            if q in unknown_verb:
                results[q] = BindingCheck(ok=False, reason=f"undefined verb: {verb}")
            elif agent in unknown_type or target in unknown_type:
                bad = agent if agent in unknown_type else target
                results[q] = BindingCheck(ok=False, reason=f"undefined type: {bad}")
            elif q in admissible:
                results[q] = BindingCheck(ok=True, rule=f"{verb}: CΩ admissible")
            elif q in violation:
                results[q] = BindingCheck(
                    ok=False,
                    reason=f"type_violation: {verb} not admissible for ({agent}, {target})",
                )
            else:  # pragma: no cover - defensive
                results[q] = BindingCheck(ok=False, reason="unresolved")
        return results

    def _write_facts(self, fdir: Path, queries: list[Query]) -> None:
        _write(fdir / "type.facts", ((name,) for name in self._onto.entities))
        _write(
            fdir / "subtype_direct.facts",
            (
                (e.name, e.subtype_of)
                for e in self._onto.entities.values()
                if e.subtype_of is not None
            ),
        )
        _write(fdir / "verb.facts", ((name,) for name in self._onto.verbs))
        _write(
            fdir / "constraint.facts",
            (
                (c.verb, c.agent_must_be, target)
                for c in self._onto.constraints
                for target in c.target_must_be
            ),
        )
        _write(fdir / "query.facts", queries)


def _write(path: Path, rows: Iterable[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for row in rows:
            writer.writerow(row)


def _read_tuples(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {tuple(row) for row in csv.reader(fh, delimiter="\t") if row}
