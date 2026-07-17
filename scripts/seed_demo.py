"""Seed the real backends with the demo world's data, so the full-stack pipeline has content.

Neo4j starts empty and Postgres has no table, so `/compile` would return an empty causal slice
and empty state until this runs. It loads the same demo data the in-memory world carries:
  - Postgres: a `gdp_state` table with one GDP row (feeds world state Eₜ).
  - Neo4j: the PolicyRate → DXY → FX_EM → GDP causal chain.

Usage (inside the API container, which has the drivers + env):
    docker compose -f infra/docker-compose.yml run --rm api python scripts/seed_demo.py
"""

from __future__ import annotations

import os

import psycopg
from loka_causal.neo4j_graph import Neo4jCausalGraph
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    IdentificationStatus,
)
from neo4j import GraphDatabase


def seed_postgres(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS gdp_state")
        conn.execute("CREATE TABLE gdp_state (id text, value double precision, unit text)")
        conn.execute("INSERT INTO gdp_state VALUES ('TH', 2.1, 'pct_yoy')")
    print("[seed] postgres: gdp_state ready (1 row)")


def seed_neo4j(uri: str, user: str, password: str) -> None:
    # clean slate
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as s:
        s.run("MATCH (n:Var) DETACH DELETE n")
    driver.close()

    g = Neo4jCausalGraph.connect(uri, user, password)
    s = IdentificationStatus
    g.add_claim(_claim("c1", "PolicyRate", "DXY", s.STRUCTURAL, CausalLayer.STRUCTURAL))
    g.add_claim(_claim("c2", "DXY", "FX_EM", s.STRUCTURAL, CausalLayer.STRUCTURAL))
    g.add_claim(_claim("c3", "FX_EM", "GDP", s.QUASI_EXPERIMENTAL, CausalLayer.EMPIRICAL))
    g.close()
    print("[seed] neo4j: 3 causal claims ready")


def _claim(
    cid: str, cause: str, effect: str, status: IdentificationStatus, layer: CausalLayer
) -> CausalClaim:
    return CausalClaim(
        claim_id=cid,
        cause=cause,
        effect=effect,
        effect_distribution=EffectDistribution(mean=-1.0, se=0.3),
        identification_status=status,
        layer=layer,
    )


if __name__ == "__main__":
    seed_postgres(os.environ["LOKA_PG_DSN"])
    seed_neo4j(
        os.environ["NEO4J_URI"],
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "loka_password"),
    )
    print("[seed] done")
