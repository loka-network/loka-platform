# Loka Platform

**A governed, executable economic world model.** Loka turns natural-language questions
into calibrated, causally-grounded, and auditable decision support — while customer data
never leaves the customer's environment.

---

## Why

Macro-financial decisions — a central bank weighing a rate move, a finance ministry sizing
an issuance, a supervisor modelling a stress path, a sovereign fund re-hedging — are **not
forecasting problems**. They are decisions taken under uncertainty about how the world will
react to an intervention, subject to rules the institution cannot break, optimised against a
mandate only the institution can declare.

Five distinct objects are routinely conflated:

| Object | Question it answers |
| --- | --- |
| Point forecast | Expected value of *Y* at *t+h*? |
| Conditional forecast | *Y* at *t+h* given *Xₜ*? |
| Counterfactual simulation | What would *Y* be if intervention *d* were applied? |
| Policy design | Which *d* should we choose? |
| Constrained decision support | Which *d* is best under our welfare, constraints, and authority? |

Existing tools each cover part of this surface. Loka unifies **data, causal mechanisms,
institutional rules, actor behaviour, and the customer's own mandate** into one runnable
representation, and reasons on it under the audit discipline a regulator would demand.

## What makes it different

- **Causal-first, not document-first.** Every quantitative claim resolves to a typed causal
  record with an effect distribution, identifying assumptions, and supporting evidence. A
  free-text assertion ("a rate cut weakens the currency") is not admitted unless it resolves
  to a causal claim with an admissible identification status.
- **Multi-agent simulation of named stakeholders.** Scenarios play out in a virtual
  environment populated by archetypes calibrated to real institutions. Adversarial moves are
  first-class actions, not sensitivity sweeps.
- **Constrained decision support, signed to the mandate.** Outputs are evaluated against the
  customer's signed welfare function, hard constraints, and authority graph — objects the
  model never derives on its own.
- **Auditable by construction.** Every output is signed; every claim types back to a source;
  every run is replayable from a manifest.
- **Sovereign.** Enterprise data is read through typed, read-only adapters; raw data is never
  copied into managed storage.

## How it works

```text
natural-language question
        │
        ▼
  Semantic Grounding ───────────►  typed query  q*
        │                          (admission checks)
        ▼
  World Model Compiler  ──────────►  Scenario World Model  W(q, t)
        │   binds: ontology Ω · causal Γ/Kt · live state Eₜ · mission
        ▼
  Cognitive & Decision Engine
    plan → simulate → forecast → decide
        │
        ▼
  Governed Outputs
    forecasts · scenario analysis · decision memorandum · external actions
```

`W(q, t)` is the compiled, per-question world model that every downstream component reads —
the single interface that keeps the system decoupled and replayable.

## Architecture

Loka is a set of **independently deployable logical services** with typed, versioned
interfaces:

| Service | Responsibility |
| --- | --- |
| `ontology` | Vocabulary Ω = (E, V, R, ⪯, CΩ): entity types, verbs, relations, subtyping, constraints |
| `causal` / `knowledge` | Causal mechanism graph Γ and its evidence & provenance layer Kt |
| `state` | Live world state Eₜ (observations and events) |
| `compiler` | Binds assets into the Scenario World Model `W(q, t)` |
| `grounding` | Natural language → typed query `q*`; admission gate |
| `manager` / `society` / `consensus` | Planning, multi-agent simulation, agent consensus |
| `gates` | Governance: admission / runtime / decision / review |
| `serving` | Model inference endpoints |

Model **training** (forecasting and decision models) lives in the separate `loka-models`
repository and integrates through a model registry.

## Repository layout

```text
libs/loka-schemas/     Shared contracts (typed schemas / proto)
services/
  ontology/            Ontology engine Ω
  adapters/            Read-only typed data adapters
  state/               World-state service Eₜ
  mission/             Mission Profile
  compiler/            World Model Compiler → W(q, t)
  causal/              Causal knowledge graph Γ
  knowledge/           Evidence & provenance layer Kt
  causal_pipeline/     Causal construction (extract / synthesise / review)
  grounding/           Semantic grounding & admission
  manager/             Planning
  society/             Multi-agent simulation
  consensus/           Agent consensus
  gates/               Governance gates
  serving/             Model serving endpoints
storage/               Unified typed query layer
infra/                 Deployment / CI
tests/                 Integration tests
```

## Governance & deployment

- **Four gates** enforce declared properties at concrete handoffs: **admission** (typing &
  evidence sufficiency), **runtime** (budget, calibration, safety), **decision** (hard
  constraints, authority, evidence grounding), **review** (gated, human-approved updates).
  No LLM sits on a gate's critical path.
- **Five deployment modes**, from managed SaaS to single-tenant, customer VPC, hybrid, and
  fully **air-gapped / sovereign** — the same typed interfaces behind each; only the location
  of data and the network boundary change.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "services/ontology[dev]"

pytest services/ontology -v
mypy --strict services/ontology/loka_ontology
ruff check services/ontology
```

## Engineering principles

1. Every module is an independently deployable service (own container, own API).
   A monorepo is not a monolith.
2. Shared contracts live in `libs/loka-schemas`; every service depends on them.
3. Services communicate only through contracts and public APIs — never by importing another
   service's internals.
4. Models are loaded from the registry by content hash.

## Status

Early development.

## License

Proprietary — Loka Labs. All rights reserved.
