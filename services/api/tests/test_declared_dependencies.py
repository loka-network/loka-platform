"""Every loka package the API imports must be declared as a dependency.

The test suite installs all ten packages, so an undeclared dependency stays invisible here and
fails only for someone who installs the API the way its own metadata describes. That is the
worst place to find it: the suite is green and the install is broken. This test reads the
imports out of the source and checks them against the packaging metadata, so the two cannot
drift apart again.

An import may be declared as required or as an extra — ``loka_adapters`` is only reached when a
Postgres DSN is configured, and forcing that on an in-memory install would be wrong. What is not
allowed is being absent from both.
"""

from __future__ import annotations

import ast
import pathlib
import re

_API = pathlib.Path(__file__).resolve().parents[1]
_PKG = _API / "loka_api"


def _imported_loka_packages() -> set[str]:
    """Top-level ``loka_*`` modules imported anywhere in the package, including inside
    functions — a deferred import fails just as hard, only later."""
    found: set[str] = set()
    for path in _PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found if m.startswith("loka_")}


def _declared_distributions() -> set[str]:
    """Distribution names from pyproject, normalised to import names (``loka-x`` -> ``loka_x``).

    Both ``dependencies`` and every ``optional-dependencies`` group count as declared.
    """
    text = (_API / "pyproject.toml").read_text()
    names = re.findall(r'"(loka-[a-z-]+)[^"]*"', text)
    return {n.replace("-", "_") for n in names}


def test_every_imported_loka_package_is_declared() -> None:
    missing = _imported_loka_packages() - _declared_distributions()
    assert not missing, (
        f"loka_api imports {sorted(missing)} but services/api/pyproject.toml does not declare "
        "them; a clean install would raise ModuleNotFoundError at runtime"
    )


def test_the_check_can_actually_fail() -> None:
    """A guard that cannot fail guards nothing: confirm both sides are non-empty and that an
    undeclared name would be reported."""
    imported = _imported_loka_packages()
    assert "loka_knowledge" in imported, "expected the Kt import that this test was written for"
    assert _declared_distributions(), "no loka distributions parsed out of pyproject.toml"
    assert {"loka_not_a_package"} - _declared_distributions() == {"loka_not_a_package"}
