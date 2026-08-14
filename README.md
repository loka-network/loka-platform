# Loka Platform

> _A governed, executable economic world model — it turns a question into a calibrated,_
> _causally-grounded, auditable decision._

Loka is a governed, executable, continuously-updated representation of the macro-financial
economy. It unifies typed world state, causal mechanisms, institutional rules, actor
behaviour, enterprise knowledge, and human objectives into one runnable model — enabling
counterfactual simulation, calibrated forecasting, and auditable decision support. Customer
data never leaves the customer's environment.

> **What this document is.** The sections below describe the design the platform is being built
> toward. Parts of it are running and tested; parts are not yet written. Rather than qualify
> every sentence, the split is stated once, here:
>
> **Built and tested** — the ontology Ω with its consistency rules; query grounding and typed
> refusal; relation traversal (routes derived from Ω, not hand-written joins); the ontology
> review lifecycle (draft → validated → published); provenance separation of observations from
> computed values; the replayable audit hash; the provider-agnostic model gateway; read-only
> data adapters.
>
> **Designed, not built** — planning; the multi-agent simulation of named stakeholders; the
> four governance gates as a distinct layer; the semantic remainder of CΩ; execution of an
> action's effect against world state.
>
> `docs/loka_technical_design.md` carries the detail, including what each guarantee rests on
> and what is explicitly *not* guaranteed.

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
  record with an effect distribution, an identification status, and evidence; a free-text
  assertion is never admitted. An admissibility matrix is implemented in `services/causal` and
  is not yet wired into the query path.
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
| `serving` | Model gateway (provider-agnostic) and the behavior-engine port |
| `api` | HTTP surface and orchestration |

The mission profile — the customer-signed mandate, welfare, constraints and authority — is a
contract in `libs/loka-schemas`; it has no service of its own yet.

Model **training** (forecasting and decision models) lives in the separate `loka-models`
repository and integrates through a model registry.

## Repository layout

```text
libs/loka-schemas/     Shared contracts and the protocols services are written against
services/
  ontology/            Ontology engine Ω, its consistency rules, route search and traversal
  adapters/            Read-only typed data adapters
  state/               World-state service Eₜ
  compiler/            World Model Compiler → W(q, t)
  causal/              Causal knowledge graph Γ + admissibility
  knowledge/           Evidence & provenance layer Kt
  grounding/           Semantic grounding & admission
  serving/             Model gateway and the behavior-engine port
  api/                 HTTP surface and orchestration
infra/                 Deployment / CI
examples/              Ontologies, data-build scripts, runnable demos

Directories reserved for services that are designed but not yet written — `manager/`,
`society/`, `consensus/`, `gates/`, `mission/`, `causal_pipeline/`, `storage/` — are empty
and are listed here so a name in the tree is never mistaken for a component.
```

## Governance & deployment

- **Four gates** are designed to enforce declared properties at concrete handoffs — admission,
  runtime, decision, review — with no model on any gate's critical path. What runs today is the
  admission check (grounding refuses a query Ω cannot type) and the review gate (an ontology
  authorises nothing until a person publishes it); the other two are not yet a distinct layer.
- **Five deployment modes**, from managed SaaS to single-tenant, customer VPC, hybrid, and
  fully air-gapped / sovereign. The same typed interfaces sit behind each; only the location
  of data and the network boundary change.

## Development

Python 3.11+, `mypy --strict`, `ruff`. Each service is an installable package.

```bash
python -m venv .venv && source .venv/bin/activate
for pkg in libs/loka-schemas services/adapters services/ontology \
           services/state services/causal services/knowledge services/compiler \
           services/grounding services/serving services/api; do
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

The loader's eight consistency rules are the checker that runs. A Soufflé/Datalog path exists
(`souffle_checker.py`) for the semantic rules beyond them; those rules are not yet written, so
nothing on the query path calls it.

Integration tests for the production backends are skipped unless a database is reachable:

```bash
docker compose -f infra/docker-compose.yml up -d
export LOKA_PG_DSN="postgresql://loka:loka@localhost:5432/loka"
export NEO4J_URI="bolt://localhost:7687" NEO4J_USER=neo4j NEO4J_PASSWORD=loka_password
pytest libs services -v     # backend integration tests now run
```

## HTTP API

A minimal FastAPI service exposes the foundation. It ships a zero-config in-memory world, so
it runs immediately (production wires real backends in `loka_api.world`).

```bash
pip install -e "services/api[dev]"
uvicorn loka_api.app:app --reload          # http://localhost:8000/docs

curl localhost:8000/health
curl -X POST localhost:8000/compile -H 'content-type: application/json' \
     -d '{"query_id":"q1","task_type":"counterfactual","targets":["GDP"]}'
# → the compiled W(q, t): state slice, causal slice Γ(q), welfare, constraints, manifest pins
```

Or run the whole stack (API + Postgres + Neo4j + Redis) with Docker:

```bash
docker compose -f infra/docker-compose.yml up -d --build   # API on :8000
```

### Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness + ontology version |
| `POST /compile` | typed query q* → W(q, t) |
| `POST /ask` | natural-language question → typed query → answer, or a typed refusal |
| `POST /project` | apply the projection method directly (no model call) |
| `GET /scenario` | the method's fields and the Ω attributes they are bound to |
| `GET /kb` | observed facts and, separately, computed counterfactual ones |
| `POST /build-kb` | domain text → draft ontology + review checklist |
| `GET /ontology`, `GET\|PUT /ontology/{id}`, `POST /ontology/{id}/publish` | review lifecycle |
| `GET /supply/scenario` | entities, relations with their link fields, guarded actions |
| `GET /supply/route` | the route Ω declares between two entity types |
| `POST /supply/impact` | tighten an action's guard → what loses eligibility, what it reaches |
| `POST /answer` | the full chain (grounding → W(q,t) → simulate → policy) |
| `POST /compile-ontology` | compile an externally-authored ontology into W(q, t) |
| `POST /kb/{id}/ingest` | fill a built KB with data rows and causal claims |

The natural-language front end is built: `POST /ask` formalises a question against the ontology
and refuses it — naming the check that failed — when the ontology cannot ground it.

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
