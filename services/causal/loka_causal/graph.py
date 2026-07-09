"""Causal knowledge graph Γ — in-memory reference implementation.

Nodes are macro-financial variables; directed edges are typed causal claims. Provides the
four standard graph operations (ancestors / descendants / mediators / confounders) with
optional layer filtering, so the causal-core (structural + empirical) can be separated from
the quarantined hypothesis layer.

A production Neo4j-backed graph will implement the same query surface; this in-memory version
is dependency-free for tests and local development.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from loka_schemas import CausalClaim, CausalLayer, IdentificationStatus

CORE_LAYERS = frozenset({CausalLayer.STRUCTURAL, CausalLayer.EMPIRICAL})


class CausalError(Exception):
    """Invalid operation on the causal graph."""


class CausalGraph:
    """A directed graph of typed causal claims."""

    def __init__(self) -> None:
        self._claims: dict[str, CausalClaim] = {}

    def add_claim(self, claim: CausalClaim) -> None:
        """Add a causal edge. Simulator-derived claims are quarantined to the hypothesis layer."""
        if (
            claim.identification_status == IdentificationStatus.SIMULATOR_DERIVED
            and claim.layer != CausalLayer.HYPOTHESIS
        ):
            raise CausalError(
                "simulator-derived claims must be quarantined in the hypothesis layer"
            )
        self._claims[claim.claim_id] = claim

    def claims(self) -> tuple[CausalClaim, ...]:
        return tuple(self._claims.values())

    def nodes(self) -> set[str]:
        out: set[str] = set()
        for c in self._claims.values():
            out.add(c.cause)
            out.add(c.effect)
        return out

    # ---- adjacency (built per query with an optional layer filter) ----

    def _adjacency(
        self, *, reverse: bool, layers: frozenset[CausalLayer] | None
    ) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for c in self._claims.values():
            if layers is not None and c.layer not in layers:
                continue
            if reverse:
                adj[c.effect].add(c.cause)
            else:
                adj[c.cause].add(c.effect)
        return adj

    @staticmethod
    def _reachable(start: str, adj: dict[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(adj.get(start, set()))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(adj.get(node, set()))
        return seen

    # ---- the four standard operations ----

    def ancestors(self, v: str, *, layers: frozenset[CausalLayer] | None = None) -> set[str]:
        """Variables that may causally affect ``v``."""
        return self._reachable(v, self._adjacency(reverse=True, layers=layers))

    def descendants(self, u: str, *, layers: frozenset[CausalLayer] | None = None) -> set[str]:
        """Variables ``u`` may causally affect."""
        return self._reachable(u, self._adjacency(reverse=False, layers=layers))

    def mediators(
        self, u: str, v: str, *, layers: frozenset[CausalLayer] | None = None
    ) -> set[str]:
        """Transmission nodes on paths from ``u`` to ``v``."""
        return self.descendants(u, layers=layers) & self.ancestors(v, layers=layers)

    def confounders(
        self, u: str, v: str, *, layers: frozenset[CausalLayer] | None = None
    ) -> set[str]:
        """Common causes of ``u`` and ``v``."""
        return self.ancestors(u, layers=layers) & self.ancestors(v, layers=layers)

    # ---- claim lookup ----

    def claims_between(
        self, nodes: Iterable[str], *, layers: frozenset[CausalLayer] | None = None
    ) -> tuple[CausalClaim, ...]:
        """Claims whose cause and effect both lie within ``nodes`` (optionally layer-filtered)."""
        node_set = set(nodes)
        return tuple(
            c
            for c in self._claims.values()
            if c.cause in node_set
            and c.effect in node_set
            and (layers is None or c.layer in layers)
        )
