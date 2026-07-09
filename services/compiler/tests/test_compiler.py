"""Tests for the World Model Compiler: producing W(q, t) from Ω + Eₜ + Mission."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from loka_causal import CausalGraph
from loka_compiler import MissionNotSigned, UnknownEntity, compile_wqt
from loka_ontology import OntologyEngine, load_ontology_str
from loka_schemas import (
    CausalClaim,
    CausalLayer,
    EffectDistribution,
    IdentificationStatus,
    MissionProfile,
    ScenarioWorldModel,
    TypedQuery,
    WelfareFunctional,
    WelfareTerm,
)
from loka_state import WorldState

T = datetime(2026, 1, 1, tzinfo=UTC)

ONTO = """
version: v1
entities:
  - {type: Instrument}
  - {type: CentralBank}
verbs:
  - {name: TRADE, class: factual}
"""


def make_engine() -> OntologyEngine:
    return OntologyEngine(load_ontology_str(ONTO))


def make_state() -> WorldState:
    st = WorldState()
    st.set("CentralBank.Fed.rate", 0.05, T)
    st.set("Instrument.US10Y.kind", "SovereignBond", T)
    return st


def signed_mission() -> MissionProfile:
    return MissionProfile(
        version="m1",
        mandate="price stability",
        welfare=WelfareFunctional(terms=(WelfareTerm("inflation", 1.0),)),
        signature="signed-by-customer",
    )


def query() -> TypedQuery:
    return TypedQuery(query_id="q1", task_type="conditional_forecast", targets=("CentralBank",))


def test_compile_produces_wqt() -> None:
    w = compile_wqt(make_engine(), make_state(), signed_mission(), query(), scenario_id="s1")
    assert isinstance(w, ScenarioWorldModel)
    assert w.query_id == "q1"
    # only CentralBank state made it into the slice
    assert w.state_package.state_slice == {"CentralBank.Fed.rate": 0.05}
    # causal slice is empty until S2
    assert w.causal_slice is None
    # manifest pins are populated
    assert w.manifest.omega_version == "v1"
    assert w.manifest.mission_version == "m1"
    assert len(w.manifest.et_snapshot) == 16


def test_unsigned_mission_is_rejected() -> None:
    m = MissionProfile(
        version="m1",
        mandate="x",
        welfare=WelfareFunctional(terms=()),
        signature=None,
    )
    with pytest.raises(MissionNotSigned):
        compile_wqt(make_engine(), make_state(), m, query(), scenario_id="s1")


def test_unknown_target_is_rejected() -> None:
    q = TypedQuery(query_id="q2", task_type="x", targets=("NotAnEntity",))
    with pytest.raises(UnknownEntity):
        compile_wqt(make_engine(), make_state(), signed_mission(), q, scenario_id="s1")


def test_compile_is_deterministic() -> None:
    # same inputs → equal W(q, t) (basis for replay)
    a = compile_wqt(make_engine(), make_state(), signed_mission(), query(), scenario_id="s1")
    b = compile_wqt(make_engine(), make_state(), signed_mission(), query(), scenario_id="s1")
    assert a == b


def test_compile_fills_causal_slice_when_graph_supplied() -> None:
    g = CausalGraph()
    g.add_claim(
        CausalClaim(
            claim_id="e1",
            cause="PolicyRate",
            effect="CentralBank",
            effect_distribution=EffectDistribution(mean=-1.0, se=0.2),
            identification_status=IdentificationStatus.STRUCTURAL,
            layer=CausalLayer.STRUCTURAL,
        )
    )
    w = compile_wqt(
        make_engine(), make_state(), signed_mission(), query(), scenario_id="s1", causal=g
    )
    assert w.causal_slice is not None
    assert any(c.claim_id == "e1" for c in w.causal_slice.claims)
