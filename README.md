# Loka Platform

> _A governed, executable economic world model — it turns a question into a calibrated,_
> _causally-grounded, auditable decision._

Loka is a governed, executable, continuously-updated representation of the macro-financial
economy. It unifies typed world state, causal mechanisms, institutional rules, actor
behaviour, enterprise knowledge, and human objectives into one runnable model — enabling
counterfactual simulation, calibrated forecasting, and auditable decision support. Customer
data never leaves the customer's environment.

## The problem

Macro-financial decisions are not forecasting problems. A central bank weighing a rate move,
a finance ministry sizing an issuance, a supervisor modelling a stress path — each asks how
the world will react to an intervention, subject to rules it cannot break, optimised against
a mandate only it can declare. Five distinct objects are routinely conflated:

| Object | Question |
| --- | --- |
| Point forecast | Expected value of _Y_ at _t + h_? |
| Conditional forecast | _Y_ at _t + h_ given _Xₜ_? |
| Counterfactual simulation | What would _Y_ be if intervention _d_ were applied? |
| Policy design | Which _d_ should we choose? |
| Constrained decision support | Which _d_ is best under our welfare, constraints, and authority? |

Existing tools cover fragments of this surface. Loka covers all five on one representation,
under the audit discipline a regulator would demand of a published policy memo.

## Approach

- **Causal-first, not document-first.** Every quantitative claim resolves to a typed causal
  record with an effect distribution, an identification status, and evidence. Downstream use
  is gated by an explicit admissibility matrix; a free-text assertion is never admitted.
- **Multi-agent simulation of named stakeholders.** Scenarios play out in a virtual
  environment of archetypes calibrated to real institutions; adversarial moves are
  first-class.
- **Constrained decision support, signed to the mandate.** Recommendations are evaluated
  against the customer's signed welfare function, hard constraints, and authority graph —
  objects the model never derives on its own.
- **Auditable by construction.** Every output is signed, every claim types back to a source,
  and every run is replayable from a version-pinned manifest.
- **Sovereign.** Enterprise data is read through typed, read-only adapters; raw data is never
  copied into managed storage.

## How it works

```text
natural-language question
        │
        ▼
  Semantic Grounding ───────────►  typed query  q*     (admission checks)
        │
        ▼
  World Model Compiler ──────────►  Scenario World Model  W(q, t)
        │   binds: ontology Ω · causal Γ / Kt · live state Eₜ · signed mission
        ▼
  Cognitive & Decision Engine
        plan → simulate → forecast → decide
        │
        ▼
  Governed Outputs
        forecasts · scenario analysis · decision memorandum · external actions
```

`W(q, t)` is the compiled, per-question world model that every downstream component reads —
the single interface that keeps the system decoupled and every run reproducible.

## Core concepts

| Symbol | Meaning |
| --- | --- |
| `Ω` | Ontology — entity types, verbs, relations, subtyping, typing constraints |
| `Γ` / `Kt` | Causal mechanism graph and its evidence & provenance layer |
| `Eₜ` | Live world state (observations and events) |
| `q*` | The typed, signed query — no free text reaches the engine |
| `W(q, t)` | The compiled Scenario World Model bound for one question |
| `G1…G4` | Governance gates: admission · runtime · decision · review |

## Architecture

Independently deployable logical services with typed, versioned interfaces:

| Service | Responsibility |
| --- | --- |
| `ontology` | The vocabulary `Ω` and its type checker |
| `causal` / `knowledge` | Causal graph `Γ`, graph queries, admissibility; evidence layer `Kt` |
| `state` | Live world state `Eₜ`; ingestion from read-only adapters |
| `adapters` | Read-only, scope-bound data access (data stays in place) |
| `compiler` | Binds `Ω + Eₜ + Γ(q) + mission` into `W(q, t)` |
| `mission` | The customer-signed mandate, welfare, constraints, authority |

Model **training** (forecasting and decision models) lives in the separate `loka-models`
repository and integrates through a model registry.

## Repository layout

```text
libs/loka-schemas/     Shared contracts (typed data, adapter, mission, causal, W(q,t))
services/
  ontology/            Ontology engine Ω
  adapters/            Read-only typed data adapters
  state/               World-state service Eₜ
  mission/             Mission Profile
  compiler/            World Model Compiler → W(q, t)
  causal/              Causal knowledge graph Γ + admissibility
  knowledge/           Evidence & provenance layer Kt
  grounding/           Semantic grounding & admission
  manager/ society/ consensus/   Planning, simulation, consensus
  gates/ serving/      Governance gates; model serving endpoints
storage/               Unified typed query layer
infra/                 Deployment / CI
```

## Governance & deployment

- **Four gates** enforce declared properties at concrete handoffs — admission, runtime,
  decision, review — with no LLM on any gate's critical path.
- **Five deployment modes**, from managed SaaS to single-tenant, customer VPC, hybrid, and
  fully air-gapped / sovereign. The same typed interfaces sit behind each; only the location
  of data and the network boundary change.

## Development

Python 3.11+, `mypy --strict`, `ruff`. Each service is an installable package.

```bash
python -m venv .venv && source .venv/bin/activate
for pkg in libs/loka-schemas services/adapters services/ontology \
           services/state services/causal services/knowledge services/compiler; do
  pip install -e "$pkg[dev]"
done

ruff check libs services
pytest libs services -v
```

### Backends (swappable via ports)

Each engine ships an in-memory reference implementation (used by the unit tests) and an
optional production backend selected behind the same port:

| Port | Reference | Production backend | Extra |
| --- | --- | --- | --- |
| data adapter | `InMemoryAdapter` | `PostgresAdapter` | `services/adapters[postgres]` |
| causal graph | `CausalGraph` | `Neo4jCausalGraph` | `services/causal[neo4j]` |

Full-strength CΩ type checking uses Soufflé when the `souffle` binary is present (the
pure-Python checker is the default).

Integration tests for the production backends are skipped unless a database is reachable:

```bash
docker compose -f infra/docker-compose.yml up -d
export LOKA_PG_DSN="postgresql://loka:loka@localhost:5432/loka"
export NEO4J_URI="bolt://localhost:7687" NEO4J_USER=neo4j NEO4J_PASSWORD=loka_password
pytest libs services -v     # backend integration tests now run
```

## Engineering principles

1. Every module is an independently deployable service. A monorepo is not a monolith.
2. Shared contracts live in `libs/loka-schemas`; every service depends on them.
3. Services communicate only through contracts and public APIs — never another service's
   internals.
4. Models are loaded from the registry by content hash.

## Status

Under active development.

## License

Proprietary — Loka Labs. All rights reserved.
