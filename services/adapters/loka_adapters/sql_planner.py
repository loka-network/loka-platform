"""Constrained SQL planning — the safe core of "text-to-SQL".

Free-form LLM text-to-SQL is unreliable and unsafe. Here the SQL is *generated from the
ontology*, not written by a model: the caller passes the backing table and the column names
taken from an entity type's properties, plus optional equality filters. Every identifier is
validated (must be a plain SQL identifier) and every filter value is bound as a parameter, so
the result is always a valid, injection-safe ``SELECT``. The LLM (grounding) only decides
*which entity* to fetch; the query itself is deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SqlPlanError(ValueError):
    """A table or column name is not a safe SQL identifier."""


def _check_ident(name: str, kind: str) -> str:
    if not _IDENT.match(name):
        raise SqlPlanError(f"{kind} {name!r} is not a valid SQL identifier")
    return name


def plan_select(
    table: str,
    columns: Sequence[str],
    *,
    filters: Mapping[str, object] | None = None,
    limit: int | None = None,
) -> tuple[str, list[object]]:
    """Build a parameterised ``SELECT``. Returns ``(sql, params)``.

    ``table`` and every column/filter key must be plain identifiers (validated against the
    ontology's names upstream); filter *values* are returned as bound parameters, never
    interpolated. ``limit`` must be a non-negative int.
    """
    if not columns:
        raise SqlPlanError("at least one column is required")
    _check_ident(table, "table")
    cols = ", ".join(_check_ident(c, "column") for c in columns)
    sql = f"SELECT {cols} FROM {table}"  # noqa: S608 — identifiers validated above
    params: list[object] = []
    if filters:
        clauses = []
        for key, value in filters.items():
            clauses.append(f"{_check_ident(key, 'filter column')} = %s")
            params.append(value)
        sql += " WHERE " + " AND ".join(clauses)
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise SqlPlanError(f"limit must be a non-negative int, got {limit!r}")
        sql += f" LIMIT {limit}"
    return sql, params
