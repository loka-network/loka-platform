"""Build the causal slice Γ(q) that the compiler binds into W(q, t).

Thin convenience wrapper around ``CausalGraph.build_slice`` (kept as a standalone function
for call sites that prefer it).
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
    return graph.build_slice(targets, layers=layers)
