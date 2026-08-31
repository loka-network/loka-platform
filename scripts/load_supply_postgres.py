"""Load the supply-chain data into Postgres, with the schema taken from the ontology.

Until now the whitelisted SQL planner had never sent a statement to a database. The plan was
built, the identifiers were checked against Ω and the values were bound — and then the rows came
from CSV files read into memory. A planner that has never executed is a planner whose claim to
be safe rests entirely on reading it.

The schema is not written here. Each table's name is the ``backing`` its entity declares, each
column is an attribute Ω declares, and each column type is that attribute's base type. So a
table this creates is one ``plan_select`` can address by construction: if Ω does not declare it,
there is no column for it to name, and if Ω declares it, the column exists.

    docker compose -f infra/docker-compose.yml up -d postgres
    export LOKA_PG_DSN="postgresql://loka:loka@localhost:5432/loka"
    python scripts/load_supply_postgres.py

    python scripts/load_supply_postgres.py --check     # plan and execute, without loading

``--check`` is the part worth running twice: it plans a query through the same code path the API
uses, executes it, and prints the statement, the bound values and the rows that came back.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for pkg in ("ontology", "adapters"):
    sys.path.insert(0, str(ROOT / "services" / pkg))

from loka_ontology.loader import load_ontology  # noqa: E402
from loka_ontology.model import BaseType, Ontology  # noqa: E402

ONTOLOGY = ROOT / "examples" / "supply_ontology.yaml"
DATA = ROOT / "examples" / "supply_sample"

#: Ω's base types, in the column types a relational store uses for them. TEXT rather than
#: VARCHAR(n) on purpose: Ω states that an attribute is a string, not how long it may be, and
#: inventing a length here would put a constraint in the database that the ontology never made.
_SQL_TYPE = {
    BaseType.STRING: "text",
    BaseType.INTEGER: "bigint",
    BaseType.DOUBLE: "double precision",
    BaseType.BOOLEAN: "boolean",
    BaseType.TIMESTAMP: "timestamptz",
    BaseType.DATE: "date",
}


def _tables(onto: Ontology) -> list[tuple[str, str, list[tuple[str, str, bool]]]]:
    """(entity, table, [(column, sql type, required)]) for every entity that declares a table."""
    out = []
    for e in onto.entities.values():
        if not e.backing:
            continue
        cols = [(p.name, _SQL_TYPE[p.base_type], p.required) for p in e.properties]
        out.append((e.name, e.backing, cols))
    return out


def _coerce(value: str, sql_type: str) -> Any:
    if value == "":
        return None
    if sql_type == "bigint":
        return int(float(value))
    if sql_type == "double precision":
        return float(value)
    if sql_type == "boolean":
        return value.strip().lower() in ("1", "true", "t", "yes")
    return value


def load(dsn: str) -> None:
    import psycopg

    onto = load_ontology(ONTOLOGY)
    with psycopg.connect(dsn, autocommit=True) as conn:
        for entity, table, cols in _tables(onto):
            path = DATA / f"{entity}.csv"
            if not path.exists():
                print(f"{entity:14s} no {path.name}; skipped")
                continue

            ddl = ", ".join(
                f'"{n}" {t}{" not null" if req else ""}' for n, t, req in cols
            )
            conn.execute(f'drop table if exists "{table}"')
            conn.execute(f'create table "{table}" ({ddl})')

            with path.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                # Only the columns Ω declares. A CSV may carry more; loading them would put data
                # in the database that no query can reach and no rule can check, which is the
                # quiet version of an ontology that does not describe its own store.
                extra = set(reader.fieldnames or []) - {n for n, _, _ in cols}
                rows = [
                    [_coerce(row.get(n, ""), t) for n, t, _ in cols] for row in reader
                ]

            placeholders = ", ".join(["%s"] * len(cols))
            names = ", ".join(f'"{n}"' for n, _, _ in cols)
            with conn.cursor() as cur:
                cur.executemany(
                    f'insert into "{table}" ({names}) values ({placeholders})', rows
                )
            note = f"  (ignored {', '.join(sorted(extra))})" if extra else ""
            print(f"{entity:14s} -> {table:16s} {len(rows):>6,} rows{note}")


def check(dsn: str) -> None:
    """Plan a query the way the API does, run it, and show every step."""
    import psycopg
    from loka_adapters.sql_planner import plan_select

    onto = load_ontology(ONTOLOGY)
    seller = onto.entities["Seller"]
    columns = [p.name for p in seller.properties]

    sql, params = plan_select(
        seller.backing,
        columns,
        filters={"seller_state": "SP"},
        ranges=[("on_time_rate", "<", 0.8)],
        limit=5,
    )
    print("entity      : Seller")
    print(f"backing     : {seller.backing}          (declared in Ω, not chosen here)")
    print(f"columns     : {', '.join(columns)}")
    print(f"\nsql         : {sql}")
    print(f"parameters  : {params}\n")

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    for row in rows:
        print("   ", row)
    print(f"\n{len(rows)} row(s) from the database.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=os.environ.get("LOKA_PG_DSN"))
    ap.add_argument("--check", action="store_true", help="plan and run a query; do not load")
    args = ap.parse_args()

    if not args.dsn:
        raise SystemExit(
            "no DSN: set LOKA_PG_DSN or pass --dsn "
            "(e.g. postgresql://loka:loka@localhost:5432/loka)"
        )
    check(args.dsn) if args.check else load(args.dsn)


if __name__ == "__main__":
    main()
