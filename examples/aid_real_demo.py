"""End-to-end demo on REAL public data: World Bank → W(q, t).

Unlike ``end_to_end_demo.py`` (which hand-authors its numbers), this pulls **live public
data** and runs the aid / grant-allocation ontology through the full foundation:

    World Bank Projects  ── funding (inputs) ──┐
    World Bank WDI       ── outcomes ──────────┤
                                               ├─► draft ontology (auto, no LLM)
    examples/aid_allocation.yaml (Ω) ──────────┤
                                               └─► ingest → compile → W(q, t)

Every number printed below is fetched from a public API — no synthetic values — so the
result is reproducible from named sources. Both APIs are open (no key required).

    python examples/aid_real_demo.py

The pulled rows are also written to ``examples/aid_real_snapshot.json`` so the exact data
used is visible and the run can be cited (or replayed offline). What this demo deliberately
stops at is W(q, t): option-ranking (which grant?) and backtesting are the next stage, not
built here.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from loka_causal import CausalGraph
from loka_compiler import compile_wqt
from loka_ontology import OntologyEngine, load_ontology
from loka_ontology.infer import infer_ontology_from_rows, to_yaml
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

NOW = datetime(2026, 3, 18, tzinfo=UTC)
HERE = Path(__file__).parent
REPO = HERE.parent
ONTOLOGY = REPO / "services" / "ontology" / "examples" / "aid_allocation.yaml"
SNAPSHOT = HERE / "aid_real_snapshot.json"

# Bounded case: health-sector aid to Zambia, malaria outcome.
COUNTRY = "ZM"
COUNTRY_ISO3 = "ZMB"
SECTOR = "Health"
OUTCOME_INDICATOR = "SH.STA.MALR"  # Malaria cases reported (WDI)

PROJECTS_URL = (
    "https://search.worldbank.org/api/v2/projects"
    f"?format=json&countrycode={COUNTRY}&sector={SECTOR}&rows=10"
)
WDI_URL = (
    f"https://api.worldbank.org/v2/country/{COUNTRY_ISO3}"
    f"/indicator/{OUTCOME_INDICATOR}?format=json&date=2012:2020"
)


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "loka-demo/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (trusted public API)
        return json.load(r)


def rule(title: str) -> None:
    print(f"\n{'─' * 70}\n {title}\n{'─' * 70}")


def fetch_inputs() -> list[dict[str, object]]:
    """World Bank Projects — the funding side (who funded which project, how much)."""
    data = _get(PROJECTS_URL)
    assert isinstance(data, dict)
    projects = data["projects"]
    rows: list[dict[str, object]] = []
    for pid, p in projects.items():
        amount = str(p.get("totalamt", "0")).replace(",", "") or "0"
        rows.append(
            {
                "grant_id": pid,
                "program": str(p.get("project_name", ""))[:60],
                "amount_usd": float(amount),
                "status": str(p.get("projectstatusdisplay", "")),
                "country": (p.get("countryname") or [""])[0],
            }
        )
    return rows


def fetch_outcomes() -> dict[str, float]:
    """World Bank WDI — the outcome side (malaria cases per year), for backtesting later."""
    data = _get(WDI_URL)
    assert isinstance(data, list)
    return {d["date"]: float(d["value"]) for d in data[1] if d["value"] is not None}


def main() -> None:
    print("Loka — real-data demo (World Bank public APIs) → W(q, t)")

    # ---- 1. Fetch inputs: funding ----
    rule("1 · Inputs — World Bank Projects (Zambia · Health)")
    grants = fetch_inputs()
    for g in sorted(grants, key=lambda x: -float(x["amount_usd"]))[:5]:
        print(f"  {g['grant_id']}  ${float(g['amount_usd']):>13,.0f}  "
              f"{g['status']:<8} {g['program']}")
    print(f"  … {len(grants)} projects total")

    # ---- 2. Auto-draft an ontology from the real rows (rule-based, no LLM) ----
    rule("2 · Auto-drafted ontology from the real rows (infer_ontology_from_rows)")
    draft = infer_ontology_from_rows("Grant", grants, backing="worldbank_projects")
    print(to_yaml(draft))

    # ---- 3. Fetch outcomes: the result series to backtest against ----
    rule("3 · Outcomes — WDI malaria cases, Zambia (per year)")
    outcomes = fetch_outcomes()
    for yr in sorted(outcomes):
        print(f"  {yr}: {outcomes[yr]:>13,.0f} cases")

    # ---- 4. Save the exact real data used (so the run is citable / offline-replayable) ----
    SNAPSHOT.write_text(
        json.dumps(
            {
                "fetched_from": {"projects": PROJECTS_URL, "outcomes": WDI_URL},
                "inputs": grants,
                "outcomes": outcomes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  → real data written to {SNAPSHOT.name}")

    # ---- 5. Ingest into the world state under the curated ontology ----
    rule("4 · Ingest real data into E_t (Ω = aid_allocation.yaml)")
    engine = OntologyEngine(load_ontology(ONTOLOGY))
    state = WorldState()
    top2 = sorted(grants, key=lambda x: -float(x["amount_usd"]))[:2]
    for i, g in enumerate(top2):
        tag = chr(65 + i)  # A, B — the two candidate grants to compare
        state.set(f"Program.{tag}.name", g["program"], NOW)
        state.set(f"Program.{tag}.budget_usd", float(g["amount_usd"]), NOW)
        state.set(f"Grant.{tag}.id", g["grant_id"], NOW)
        state.set(f"Grant.{tag}.amount_usd", float(g["amount_usd"]), NOW)
    latest = max(outcomes)
    state.set("Region.ZM.name", "Zambia", NOW)
    state.set("Region.ZM.economic_index", outcomes[latest], NOW)  # latest outcome level
    print(f"  Program slice : {state.slice(['Program'])}")
    print(f"  Grant slice   : {state.slice(['Grant'])}")
    print(f"  snapshot hash : {state.snapshot_hash()}")

    # ---- 6. Mission (signed) + one causal edge + q* ----
    rule("5 · Mission + causal edge + question q*")
    mission = MissionProfile(
        version="ngo-fund-v1",
        mandate="maximise health impact under a fixed grant budget (Zambia)",
        welfare=WelfareFunctional(terms=(WelfareTerm("malaria_cases_averted", 1.0),)),
        hard_constraints=(HardConstraint("budget_cap", "total grant within envelope"),),
        signature="signed-by-foundation",
    )
    # NOTE: this effect size is a placeholder. A real edge is estimated from evidence (K_t);
    # here it only demonstrates that the causal slice is bound into W(q, t).
    causal = CausalGraph()
    causal.add_claim(
        CausalClaim(
            claim_id="grant->malaria",
            cause="GrantAmount",
            effect="MalariaCases",
            effect_distribution=EffectDistribution(mean=-0.02, se=0.008),
            identification_status=IdentificationStatus.QUASI_EXPERIMENTAL,
            layer=CausalLayer.EMPIRICAL,
        )
    )
    query = TypedQuery(
        query_id="aid-zm-q1",
        task_type="counterfactual",
        targets=("Program",),
        signature="signed-by-g1",
    )
    print(f"  mandate : {mission.mandate}")
    print(f"  q*      : targets={query.targets} task={query.task_type}")

    # ---- 7. Compile → W(q, t) ----
    rule("6 · Compile → W(q, t) = bind(Ω, E_t, Mission, Γ(q))")
    wqt = compile_wqt(engine, state, mission, query, scenario_id="aid-zm", causal=causal)
    assert wqt.causal_slice is not None
    print(f"  scenario_id : {wqt.scenario_id}")
    print(f"  state slice : {dict(wqt.state_package.state_slice)}")
    for c in wqt.causal_slice.claims:
        print(f"  causal edge : {c.cause} → {c.effect} [{c.identification_status.value}]")
    print(f"  welfare     : {[(t.name, t.weight) for t in wqt.welfare.terms]}")
    print(f"  hard limits : {[h.name for h in wqt.hard_constraints]}")
    print(f"  manifest    : Ω={wqt.manifest.omega_version} "
          f"E_t={wqt.manifest.et_snapshot} M={wqt.manifest.mission_version}")

    rule("Done")
    print("  W(q, t) compiled from real public data, reproducible via the manifest pins.")
    print("  Next stage (not built): option-ranking (A vs B) and backtesting vs the WDI series.")


if __name__ == "__main__":
    main()
