"""Compile an externally-authored ontology into W(q,t).

This is what an ontology-building front end (e.g. Loka-OntoPrompt) calls: it POSTs the
entities and relations it built; this loads them into Ω + world state Eₜ and the causal engine
(causal graph Γ) in memory to produce a reproducible Scenario World Model — no storage
backend required. The causal edge is drawn from a real relation in the supplied ontology.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from loka_causal import CausalGraph
from loka_compiler import compile_wqt
from loka_ontology import (
    BaseType,
    EntityType,
    Ontology,
    OntologyEngine,
    Property,
    Relation,
)
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    HardConstraint,
    IdentificationStatus,
    MissionProfile,
    TypedQuery,
    WelfareFunctional,
    WelfareTerm,
)
from loka_state import WorldState

_T = datetime(2026, 1, 1, tzinfo=UTC)  # fixed → stable Eₜ snapshot hash across runs
# Verbs that suggest a causal reading, in the languages an authored ontology may use. The
# Chinese entries are deliberate: an ontology written in Chinese should be recognised too.
_CAUSAL_HINTS = (
    "影响", "导致", "决定",
    "affect", "cause", "impact", "lead", "drive", "produce",
)


def _slug(name: str, fallback: str) -> str:
    return (name or "").strip() or fallback


def compile_wqt_from_ontology(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    ontology_id: str,
    ontology_name: str = "ontology",
) -> dict[str, Any]:
    """entities: [{name, properties:{}}]; relations: [{source, target, type}] (names)."""
    names: list[str] = []
    et_map: dict[str, EntityType] = {}
    for i, e in enumerate(entities):
        name = _slug(str(e.get("name", "")), f"Entity{i}")
        if name in et_map:
            continue
        seen: set[str] = set()
        props: list[Property] = []
        for key in e.get("properties") or {}:
            k = str(key)
            if k and k not in seen:
                seen.add(k)
                props.append(Property(name=k, base_type=BaseType.STRING))
        et_map[name] = EntityType(name=name, properties=tuple(props))
        names.append(name)

    rels: list[Relation] = []
    for r in relations:
        src, tgt = _slug(str(r.get("source", "")), ""), _slug(str(r.get("target", "")), "")
        if src in et_map and tgt in et_map:
            rels.append(
                Relation(
                    name=_slug(str(r.get("type", "")), "relates_to"),
                    from_type=src,
                    to_type=tgt,
                    cardinality=None,  # the external format does not carry one
                )
            )

    engine = OntologyEngine(
        Ontology(version=f"{_slug(ontology_name, 'onto')}-v1", entities=et_map, relations=rels)
    )

    causal_rel = next(
        (r for r in rels if any(h in r.name.lower() or h in r.name for h in _CAUSAL_HINTS)),
        rels[0] if rels else None,
    )
    if causal_rel is not None:
        cause, effect = causal_rel.from_type, causal_rel.to_type
    else:
        cause = names[0] if names else "Entity0"
        effect = names[1] if len(names) > 1 else cause

    state = WorldState()
    cause_props = list(
        (next((e for e in entities if _slug(str(e.get("name", "")), "") == cause), {}).get(
            "properties"
        ) or {}).keys()
    )
    for tag, score in (("A", "high"), ("B", "medium")):
        state.set(f"{cause}.{tag}.candidate", tag, _T)
        if cause_props:
            state.set(f"{cause}.{tag}.{cause_props[0]}", score, _T)
    state.set(f"{effect}.observed.value", 1.0, _T)

    mission = MissionProfile(
        version="ontoprompt-demo-v1",
        mandate=f"choose the {cause} that best improves {effect}, under a budget",
        welfare=WelfareFunctional(terms=(WelfareTerm(f"{effect}_gain", 1.0),)),
        hard_constraints=(HardConstraint("budget_cap", "total spend within envelope"),),
        signature="signed-by-ontoprompt",
    )

    graph = CausalGraph()
    graph.add_claim(
        CausalClaim(
            claim_id="edge-1",
            cause=cause,
            effect=effect,
            effect_distribution=EffectDistribution(mean=0.04, se=0.015),
            identification_status=IdentificationStatus.QUASI_EXPERIMENTAL,
            layer=CausalLayer.EMPIRICAL,
        )
    )

    targets = tuple(dict.fromkeys([effect, cause]))
    query = TypedQuery(
        query_id=f"q-{ontology_id[:8]}",
        task_type="counterfactual",
        targets=targets,
        signature="signed-by-g1",
    )
    wqt = compile_wqt(
        engine, state, mission, query, scenario_id=f"scn-{ontology_id[:8]}", causal=graph
    )
    causal_slice = [
        {
            "cause": c.cause,
            "effect": c.effect,
            "identification_status": c.identification_status.value,
            "mean": c.effect_distribution.mean,
            "se": c.effect_distribution.se,
        }
        for c in (wqt.causal_slice.claims if wqt.causal_slice else [])
    ]
    return {
        "derived": {
            "entity_count": len(names),
            "relation_count": len(rels),
            "entities": names,
            "causal_edge_from_relation": (
                {"from": causal_rel.from_type, "to": causal_rel.to_type, "type": causal_rel.name}
                if causal_rel
                else None
            ),
        },
        "wqt": {
            "scenario_id": wqt.scenario_id,
            "query_id": wqt.query_id,
            "task_type": query.task_type,
            "targets": list(query.targets),
            "state_slice": dict(wqt.state_package.state_slice),
            "causal_slice": causal_slice,
            "welfare": [{"name": t.name, "weight": t.weight} for t in wqt.welfare.terms],
            "hard_constraints": [h.name for h in wqt.hard_constraints],
            "manifest": {
                "omega_version": wqt.manifest.omega_version,
                "state_snapshot_hash": wqt.manifest.et_snapshot,
                "mission_version": wqt.manifest.mission_version,
            },
        },
    }
