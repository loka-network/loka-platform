"""Integration tests for the Neo4j causal graph backend.

Skipped unless ``NEO4J_URI`` points at a reachable Neo4j (see infra/docker-compose.yml) and the
``neo4j`` driver is installed. Run these on the cloud/dev database.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("neo4j")

from loka_causal.neo4j_graph import Neo4jCausalGraph  # noqa: E402
from loka_schemas import (  # noqa: E402
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    IdentificationStatus,
)

URI = os.environ.get("NEO4J_URI")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "loka_password")
pytestmark = pytest.mark.skipif(not URI, reason="NEO4J_URI not set")

S = IdentificationStatus


def _claim(cid: str, cause: str, effect: str) -> CausalClaim:
    return CausalClaim(
        claim_id=cid,
        cause=cause,
        effect=effect,
        effect_distribution=EffectDistribution(mean=-1.0, se=0.3),
        identification_status=S.STRUCTURAL,
        layer=CausalLayer.STRUCTURAL,
    )


@pytest.fixture
def graph() -> object:
    assert URI is not None
    g = Neo4jCausalGraph.connect(URI, USER, PASSWORD)
    # clean slate
    with g._driver.session(database=g._db) as s:  # noqa: SLF001 - test cleanup
        s.run("MATCH (n:Var) DETACH DELETE n")
    yield g
    with g._driver.session(database=g._db) as s:  # noqa: SLF001
        s.run("MATCH (n:Var) DETACH DELETE n")
    g.close()


def test_ancestors_and_slice(graph: Neo4jCausalGraph) -> None:
    graph.add_claim(_claim("c1", "Fed.rate", "DXY"))
    graph.add_claim(_claim("c2", "DXY", "TH.GDP"))
    assert graph.ancestors("TH.GDP") == {"Fed.rate", "DXY"}
    sl = graph.build_slice(["TH.GDP"])
    assert {c.claim_id for c in sl.claims} == {"c1", "c2"}
