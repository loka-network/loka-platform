"""Ingest data and causal edges into a KB built via Workflow A.

Workflow A produces the ontology + DATA/METHODS *needs*; this fills those needs so the closed
loop returns substance: state values feed the ``asks``/DATA branch, and causal claims feed the
``orders``/METHODS ``causal_effect`` method. Only identified claims (structural / experimental /
quasi-experimental) enter the causal core; observational/expert claims fall to the hypothesis
layer and are honestly excluded from the core slice Γ(q).
"""

from __future__ import annotations

from collections.abc import Iterable
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
    """Add causal claims to Γ, pooling the evidence when several entries describe one edge.

    Two entries for the same cause→effect are two pieces of evidence about one claim, not two
    claims. Taking the last to arrive would silently drop the others; averaging them would hide
    the case that matters — sources that disagree by more than sampling error. Both go through
    Kt: the estimate stored in Γ is the pooled one, and a disagreement opens a contradiction
    record that :func:`contradictions_for` can surface next to any answer that rests on it.
    """
    if not claims:
        return 0
    from loka_causal import CausalGraph
    from loka_knowledge import KnowledgeBase
    from loka_schemas import EvidenceRecord, StudyDesign

    graph = world.causal if isinstance(world.causal, CausalGraph) else CausalGraph()
    kt: KnowledgeBase = getattr(world, "knowledge", None) or KnowledgeBase()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        cid = c.get("claim_id") or f"{c['cause']}->{c['effect']}"
        grouped.setdefault(cid, []).append(c)

    # Where a caller does not state the study design, take the one its identification status
    # implies rather than inventing a default: a quasi-experimental claim did not come from an
    # RCT, and calling it "observational" would understate it.
    design_for = {
        "experimental": StudyDesign.RCT,
        "quasi_experimental": StudyDesign.NATURAL_EXPERIMENT,
        "structural": StudyDesign.STRUCTURAL_MODEL,
        "definitional": StudyDesign.STRUCTURAL_MODEL,
        "institutional": StudyDesign.STRUCTURAL_MODEL,
        "simulator_derived": StudyDesign.SIMULATION,
    }

    for cid, entries in grouped.items():
        refs: list[str] = []
        for i, c in enumerate(entries):
            status_i = IdentificationStatus(c.get("identification_status", "structural"))
            # A caller's evidence reference is a citation; keep it as the record's identity so a
            # contradiction names the sources that disagree rather than positions in a list.
            given = tuple(c.get("evidence_refs", ()))
            evidence_id = str(given[0]) if given else f"{cid}#{i}"
            refs.extend(str(r) for r in given) if given else refs.append(evidence_id)
            kt.add_evidence(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    claim_id=cid,
                    source=str(c.get("source", "ingested")),
                    study_design=StudyDesign(
                        c.get("study_design")
                        or design_for.get(status_i.value, StudyDesign.OBSERVATIONAL)
                    ),
                    estimate=EffectDistribution(
                        mean=float(c["mean"]), se=float(c.get("se", 0.0))
                    ),
                    identification_status=status_i,
                    context=c.get("context"),
                )
            )
        synthesis = kt.synthesize(cid)

        first = entries[0]
        status = IdentificationStatus(first.get("identification_status", "structural"))
        graph.add_claim(
            CausalClaim(
                claim_id=cid,
                cause=first["cause"],
                effect=first["effect"],
                effect_distribution=synthesis.pooled,  # pooled, not last-write-wins
                identification_status=status,
                layer=_LAYER.get(str(status.value), CausalLayer.HYPOTHESIS),
                assumptions=tuple(first.get("assumptions", ())),
                context=first.get("context"),
                evidence_refs=tuple(refs),
            )
        )

    world.causal = graph
    world.knowledge = kt
    return len(claims)


def contradictions_for(world: World, claim_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Unresolved disagreements among the evidence behind the given claims.

    An answer that rests on a claim whose sources disagree beyond sampling error is not the same
    as one whose sources agree, and the difference is invisible in the pooled number alone.
    """
    kt = getattr(world, "knowledge", None)
    if kt is None:
        return []
    wanted = set(claim_ids)
    return [
        {
            "claim_id": c.claim_id,
            "type": str(c.disagreement_type),
            "detail": c.detail,
            "between": [c.evidence_a, c.evidence_b],
        }
        for c in kt.contradictions()
        if not c.resolved and c.claim_id in wanted
    ]
