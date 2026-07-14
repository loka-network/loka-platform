"""Neo4j-backed causal graph Γ — the production backend.

Implements the same query surface as the in-memory ``CausalGraph`` (and the ``CausalSlicer``
port), but stores claims as a property graph in Neo4j and computes ancestors / descendants via
Cypher variable-length paths. Drop-in replacement: anything typed against the port accepts it.

Requires the ``neo4j`` driver and a running Neo4j instance (see infra/docker-compose.yml).
Verify with the gated integration tests on a live database.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from loka_schemas import (
    CausalClaim,
    CausalLayer,
    CausalSlice,
    EffectDistribution,
    IdentificationStatus,
)
from neo4j import Driver, GraphDatabase

from .graph import CORE_LAYERS, CausalError


class Neo4jCausalGraph:
    """Causal graph Γ backed by Neo4j. Same interface as the in-memory reference graph."""

    def __init__(self, driver: Driver, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._db = database

    @classmethod
    def connect(
        cls, uri: str, user: str, password: str, *, database: str = "neo4j"
    ) -> Neo4jCausalGraph:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver, database=database)

    def close(self) -> None:
        self._driver.close()

    # ---- writes ----

    def add_claim(self, claim: CausalClaim) -> None:
        if (
            claim.identification_status == IdentificationStatus.SIMULATOR_DERIVED
            and claim.layer != CausalLayer.HYPOTHESIS
        ):
            raise CausalError(
                "simulator-derived claims must be quarantined in the hypothesis layer"
            )
        query = (
            "MERGE (c:Var {name: $cause}) "
            "MERGE (e:Var {name: $effect}) "
            "MERGE (c)-[r:CAUSES {claim_id: $claim_id}]->(e) "
            "SET r.status=$status, r.layer=$layer, r.mean=$mean, r.se=$se, "
            "    r.assumptions=$assumptions, r.context=$context, r.evidence_refs=$refs"
        )
        with self._driver.session(database=self._db) as session:
            session.run(
                query,
                cause=claim.cause,
                effect=claim.effect,
                claim_id=claim.claim_id,
                status=claim.identification_status.value,
                layer=claim.layer.value,
                mean=claim.effect_distribution.mean,
                se=claim.effect_distribution.se,
                assumptions=list(claim.assumptions),
                context=claim.context,
                refs=list(claim.evidence_refs),
            )

    # ---- queries ----

    def nodes(self) -> set[str]:
        with self._driver.session(database=self._db) as session:
            result = session.run("MATCH (v:Var) RETURN v.name AS name")
            return {str(record["name"]) for record in result}

    def ancestors(self, v: str, *, layers: frozenset[CausalLayer] | None = None) -> set[str]:
        return self._reachable(v, incoming=True, layers=layers)

    def descendants(self, u: str, *, layers: frozenset[CausalLayer] | None = None) -> set[str]:
        return self._reachable(u, incoming=False, layers=layers)

    def mediators(
        self, u: str, v: str, *, layers: frozenset[CausalLayer] | None = None
    ) -> set[str]:
        return self.descendants(u, layers=layers) & self.ancestors(v, layers=layers)

    def confounders(
        self, u: str, v: str, *, layers: frozenset[CausalLayer] | None = None
    ) -> set[str]:
        return self.ancestors(u, layers=layers) & self.ancestors(v, layers=layers)

    def _reachable(
        self, name: str, *, incoming: bool, layers: frozenset[CausalLayer] | None
    ) -> set[str]:
        pattern = (
            "(other:Var)-[:CAUSES*1..]->(anchor:Var {name: $name})"
            if incoming
            else "(anchor:Var {name: $name})-[:CAUSES*1..]->(other:Var)"
        )
        layer_filter = (
            "WHERE all(rel IN relationships(p) WHERE rel.layer IN $layers) "
            if layers is not None
            else ""
        )
        query = f"MATCH p={pattern} {layer_filter}RETURN DISTINCT other.name AS name"
        params: dict[str, object] = {"name": name}
        if layers is not None:
            params["layers"] = [layer.value for layer in layers]
        with self._driver.session(database=self._db) as session:
            result = session.run(query, params)
            return {str(record["name"]) for record in result}

    def claims_between(
        self, nodes: Iterable[str], *, layers: frozenset[CausalLayer] | None = None
    ) -> tuple[CausalClaim, ...]:
        node_list = list(nodes)
        layer_filter = "AND r.layer IN $layers " if layers is not None else ""
        query = (
            "MATCH (a:Var)-[r:CAUSES]->(b:Var) "
            f"WHERE a.name IN $nodes AND b.name IN $nodes {layer_filter}"
            "RETURN a.name AS cause, b.name AS effect, r AS rel"
        )
        params: dict[str, object] = {"nodes": node_list}
        if layers is not None:
            params["layers"] = [layer.value for layer in layers]
        with self._driver.session(database=self._db) as session:
            result = session.run(query, params)
            return tuple(
                _claim_from_edge(str(rec["cause"]), str(rec["effect"]), dict(rec["rel"]))
                for rec in result
            )

    def build_slice(
        self, targets: Sequence[str], *, layers: frozenset[CausalLayer] | None = CORE_LAYERS
    ) -> CausalSlice:
        relevant: set[str] = set(targets)
        for t in targets:
            relevant |= self.ancestors(t, layers=layers)
        return CausalSlice(
            targets=tuple(targets), claims=self.claims_between(relevant, layers=layers)
        )


def _claim_from_edge(cause: str, effect: str, props: dict[str, object]) -> CausalClaim:
    return CausalClaim(
        claim_id=str(props["claim_id"]),
        cause=cause,
        effect=effect,
        effect_distribution=EffectDistribution(
            mean=float(props["mean"]), se=float(props["se"])  # type: ignore[arg-type]
        ),
        identification_status=IdentificationStatus(str(props["status"])),
        layer=CausalLayer(str(props["layer"])),
        assumptions=tuple(props.get("assumptions") or ()),  # type: ignore[arg-type]
        context=props.get("context"),  # type: ignore[arg-type]
        evidence_refs=tuple(props.get("evidence_refs") or ()),  # type: ignore[arg-type]
    )
