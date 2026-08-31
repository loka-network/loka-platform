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

from loka_ontology.engine import OntologyEngine  # noqa: E402
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
    """(entity, table, [(column, sql type, required)]) for every entity that declares a table.

    Columns come from the engine's effective properties — own plus everything inherited along ⪯
    — because that is what the reader asks for. Built from an entity's own declarations instead,
    bulky_products got one column: BulkyProduct redeclares weight_g and inherits product_id,
    category and volume_cm3 from Product. The API then selected four columns from a table with
    one and every supply endpoint failed.

    The failure is worth naming rather than just fixing. Two places computed "the attributes of
    this entity" and computed it differently, which is the inconsistency an ontology exists to
    remove; the fix is not a longer list here but a single answer both sides read.
    """
    engine = OntologyEngine(onto)
    out = []
    for name in engine.entity_types():
        table = engine.backing_of(name)
        if not table:
            continue
        props = engine.properties_of(name)
        cols = [
            (n, _SQL_TYPE[props[n].base_type], props[n].required) for n in sorted(props)
        ]
        out.append((name, table, cols))
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

    _verify(dsn, onto)


def _verify(dsn: str, onto: Ontology) -> None:
    """Run the query the API will run, against every table just created.

    Loading and reading each decide for themselves what an entity's attributes are, and the
    first time they disagreed nothing noticed: bulky_products was created with one column and
    the API asked it for four, so every supply endpoint failed with a database error while the
    load had reported six tables and sixteen thousand rows. A load that reports success for a
    schema no query can use has reported the wrong thing.
    """
    import psycopg
    from loka_adapters.sql_planner import plan_select

    engine = OntologyEngine(onto)
    failures: list[str] = []
    with psycopg.connect(dsn) as conn:
        for name in engine.entity_types():
            table = engine.backing_of(name)
            if not table:
                continue
            sql, params = plan_select(table, sorted(engine.properties_of(name)), limit=1)
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    cur.fetchall()
            except Exception as exc:  # noqa: BLE001 - reported per table, not raised on the first
                conn.rollback()
                failures.append(f"{name}: {str(exc).strip().splitlines()[0]}")

    if failures:
        raise SystemExit(
            "the tables were created, but the query the API issues does not run against them:\n"
            + "\n".join(f"  {f}" for f in failures)
        )
    print("\nevery entity answers the query the API will issue for it.")


def check(dsn: str) -> None:
    """Plan a query the way the API does, run it, and show every step."""
    import psycopg
    from loka_adapters.sql_planner import plan_select

    engine = OntologyEngine(load_ontology(ONTOLOGY))
    backing = engine.backing_of("Seller")
    columns = sorted(engine.properties_of("Seller"))

    sql, params = plan_select(
        backing,
        columns,
        filters={"seller_state": "SP"},
        ranges=[("on_time_rate", "<", 0.8)],
        limit=5,
    )
    print("entity      : Seller")
    print(f"backing     : {backing}          (declared in Ω, not chosen here)")
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
