"""End-to-end demo: q* → (S1 + S2) → W(q, t).

Runnable proof that the foundation is wired together. It hand-authors a typed query (the
natural-language front-end is S3, not built yet), then binds the ontology (S1), world state
(S1), mission (S1) and causal graph (S2) into a single, reproducible Scenario World Model.

    python examples/end_to_end_demo.py

The "result" here is the compiled W(q, t) — the structured scenario package every downstream
engine reads. Turning it into a human-facing report is S3 (LLM) / S4-S6.
"""

from __future__ import annotations

from datetime import UTC, datetime

from loka_causal import CausalGraph
from loka_compiler import compile_wqt
from loka_knowledge import KnowledgeBase
from loka_ontology import OntologyEngine, load_ontology_str
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    EvidenceRecord,
    HardConstraint,
    IdentificationStatus,
    MissionProfile,
    StudyDesign,
    TypedQuery,
    WelfareFunctional,
    WelfareTerm,
)
from loka_state import WorldState

NOW = datetime(2026, 3, 18, tzinfo=UTC)

ONTOLOGY = """
version: demo-v1
entities:
  - type: MacroIndicator
    properties:
      - {name: unit, type: string}
  - type: GDP
    subtype_of: MacroIndicator
    properties:
      - {name: value, type: double, required: true}
  - {type: CentralBank}
  - {type: PolicyLever}
verbs:
  - {name: RATE_CHANGE, class: institutional}
relations:
  - {name: sets, from: CentralBank, to: PolicyLever, cardinality: one_to_many}
"""


def rule(title: str) -> None:
    print(f"\n{'─' * 68}\n {title}\n{'─' * 68}")


def main() -> None:
    print("Loka end-to-end demo — q* → (S1 + S2) → W(q, t)")

    # ---- S1: ontology Ω ----
    rule("S1 · Ontology Ω")
    engine = OntologyEngine(load_ontology_str(ONTOLOGY))
    print(f" entity types : {sorted(engine.entity_types())}")
    print(f" GDP ⪯ MacroIndicator : {engine.is_subtype('GDP', 'MacroIndicator')}")
    print(f" GDP properties (inherited): {sorted(engine.properties_of('GDP'))}")

    # ---- S1: world state Eₜ ----
    rule("S1 · World state Eₜ")
    state = WorldState()
    state.set("GDP.TH.value", 2.1, NOW)
    state.set("GDP.TH.unit", "pct_yoy", NOW)
    state.set("CentralBank.Fed.policy_rate", 0.0525, NOW)
    print(f" GDP slice     : {state.slice(['GDP'])}")
    print(f" snapshot hash : {state.snapshot_hash()}")

    # ---- S1: signed mission profile ----
    rule("S1 · Mission Profile (customer-signed)")
    mission = MissionProfile(
        version="ministry-v1",
        mandate="imported-inflation moderation with output-gap secondary",
        welfare=WelfareFunctional(
            terms=(WelfareTerm("inflation_dev", 0.7), WelfareTerm("output_gap", 0.3))
        ),
        hard_constraints=(HardConstraint("no_capital_controls", "jurisdiction forbids them"),),
        signature="signed-by-ministry",
    )
    print(f" mandate  : {mission.mandate}")
    print(f" welfare  : {[(t.name, t.weight) for t in mission.welfare.terms]}")
    print(f" signed   : {mission.is_signed}")

    # ---- S2: causal graph Γ ----
    rule("S2 · Causal graph Γ")
    graph = CausalGraph()
    S = IdentificationStatus
    graph.add_claim(_claim("c1", "PolicyRate", "DXY", S.STRUCTURAL, CausalLayer.STRUCTURAL))
    graph.add_claim(_claim("c2", "DXY", "FX_EM", S.STRUCTURAL, CausalLayer.STRUCTURAL))
    graph.add_claim(_claim("c3", "FX_EM", "GDP", S.QUASI_EXPERIMENTAL, CausalLayer.EMPIRICAL))
    print(f" ancestors(GDP) : {sorted(graph.ancestors('GDP'))}")
    print(f" mediators(PolicyRate → GDP) : {sorted(graph.mediators('PolicyRate', 'GDP'))}")

    # ---- S2: Kt evidence synthesis for one claim ----
    rule("S2 · Kt evidence synthesis (claim c3)")
    kb = KnowledgeBase()
    for i, m in enumerate([-0.8, -1.0, -1.1]):
        kb.add_evidence(_evidence(f"e{i}", "c3", m))
    syn = kb.synthesize("c3")
    lo, hi = syn.pooled.ci95
    print(f" pooled effect : {syn.pooled.mean:.3f}  (95% CI {lo:.2f}..{hi:.2f})")
    print(f" heterogeneity : I²={syn.i_squared:.2f}  contradictions={len(syn.contradictions)}")

    # ---- the question (hand-authored q*; NL→q* is S3) ----
    rule("Question q* (hand-authored — NL→q* is S3)")
    query = TypedQuery(
        query_id="demo-q1",
        task_type="counterfactual",
        targets=("GDP",),
        signature="signed-by-g1",
    )
    print(" 'If the Fed surprises with a rate change, what happens to GDP?'")
    print(f" q* targets : {query.targets}   task : {query.task_type}")

    # ---- COMPILE: bind everything into W(q, t) ----
    rule("Compiler · W(q, t) = bind(Ω, Eₜ, Mission, Γ(q))")
    wqt = compile_wqt(engine, state, mission, query, scenario_id="demo-s1", causal=graph)
    print(f" scenario_id   : {wqt.scenario_id}")
    print(f" state slice   : {dict(wqt.state_package.state_slice)}")
    assert wqt.causal_slice is not None
    print(" causal slice Γ(q):")
    for c in wqt.causal_slice.claims:
        eff = c.effect_distribution
        print(f"   {c.cause:>10} → {c.effect:<6} [{c.identification_status.value}] "
              f"mean={eff.mean:+.2f} se={eff.se:.2f}")
    print(f" welfare terms : {[(t.name, t.weight) for t in wqt.welfare.terms]}")
    print(f" hard limits   : {[h.name for h in wqt.hard_constraints]}")
    print(f" manifest pins : Ω={wqt.manifest.omega_version} "
          f"Eₜ={wqt.manifest.et_snapshot} Mc={wqt.manifest.mission_version}")

    rule("Done")
    print(" W(q, t) compiled and reproducible. Downstream (S3 LLM / S4-S6) turns this")
    print(" structured package into a human-facing report — not built yet.")


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


def _evidence(eid: str, claim_id: str, mean: float) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        claim_id=claim_id,
        source=f"study-{eid}",
        study_design=StudyDesign.DID,
        estimate=EffectDistribution(mean=mean, se=0.3),
        identification_status=IdentificationStatus.QUASI_EXPERIMENTAL,
    )


if __name__ == "__main__":
    main()
