"""Build the causal slice Γ(q) that the compiler binds into W(q, t).

For a question's target variables, collect the relevant causal subgraph: each target, its
ancestors (the conditioning set), and the claims among them. By default only the causal core
(structural + empirical layers) is included; the quarantined hypothesis layer is excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

from loka_schemas import CausalLayer, CausalSlice

from .graph import CORE_LAYERS, CausalGraph


def build_slice(
    graph: CausalGraph,
    targets: Sequence[str],
    *,
    layers: frozenset[CausalLayer] | None = CORE_LAYERS,
) -> CausalSlice:
    """Extract Γ(q): targets + their ancestors + the claims among them."""
    relevant: set[str] = set(targets)
    for t in targets:
        relevant |= graph.ancestors(t, layers=layers)
    claims = graph.claims_between(relevant, layers=layers)
    return CausalSlice(targets=tuple(targets), claims=claims)
