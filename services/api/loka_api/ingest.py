"""Ingest data and causal edges into a KB built via Workflow A.

Workflow A produces the ontology + DATA/METHODS *needs*; this fills those needs so the closed
loop returns substance: state values feed the ``asks``/DATA branch, and causal claims feed the
``orders``/METHODS ``causal_effect`` method. Only identified claims (structural / experimental /
quasi-experimental) enter the causal core; observational/expert claims fall to the hypothesis
layer and are honestly excluded from the core slice Γ(q).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from loka_schemas import CausalClaim, CausalLayer, EffectDistribution, IdentificationStatus

from .world import World

_T = datetime(2026, 1, 1, tzinfo=UTC)

# identification status -> which causal layer the claim lands in
_LAYER = {
    "structural": CausalLayer.STRUCTURAL,
    "definitional": CausalLayer.STRUCTURAL,
    "institutional": CausalLayer.STRUCTURAL,
    "experimental": CausalLayer.EMPIRICAL,
    "quasi_experimental": CausalLayer.EMPIRICAL,
}


def ingest_data(world: World, entries: list[dict[str, Any]]) -> int:
    """Set state values from ``{entity, instance, property, value}`` entries (KB.DATA)."""
    n = 0
    for e in entries:
        key = f"{e['entity']}.{e.get('instance', '0')}.{e['property']}"
        world.state.set(key, e["value"], _T)
        n += 1
    return n


def ingest_causal(world: World, claims: list[dict[str, Any]]) -> int:
    """Add causal claims to Γ from ``{cause, effect, mean, se, identification_status}`` entries."""
    if not claims:
        return 0
    from loka_causal import CausalGraph

    graph = world.causal if isinstance(world.causal, CausalGraph) else CausalGraph()
    n = 0
    for c in claims:
        status = IdentificationStatus(c.get("identification_status", "structural"))
        layer = _LAYER.get(str(status.value), CausalLayer.HYPOTHESIS)
        graph.add_claim(
            CausalClaim(
                claim_id=c.get("claim_id") or f"{c['cause']}->{c['effect']}",
                cause=c["cause"],
                effect=c["effect"],
                effect_distribution=EffectDistribution(
                    mean=float(c["mean"]), se=float(c.get("se", 0.0))
                ),
                identification_status=status,
                layer=layer,
                assumptions=tuple(c.get("assumptions", ())),
                context=c.get("context"),
                evidence_refs=tuple(c.get("evidence_refs", ())),
            )
        )
        n += 1
    world.causal = graph
    return n
