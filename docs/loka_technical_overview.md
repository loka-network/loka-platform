# Loka — Technical Overview

> A working technical document for the Loka platform. **Part I** introduces the system and its
> mechanisms independently of any single demo; **Part II** walks through one applied scenario end
> to end. Implementation status is marked throughout: ✅ implemented · 🟡 basic / stand-in · 🔴 not
> yet built — with the reason stated, so nothing is oversold.

---

## 0. Access & environment

- **Repository:** https://github.com/loka-network/loka-platform
- **Branch:** `feat/agent-skeleton`
- **Test API base URL:** `http://149.102.145.51:8100`

Quick calls:

```
curl http://149.102.145.51:8100/health
curl http://149.102.145.51:8100/scenario
curl -X POST http://149.102.145.51:8100/project \
  -H 'Content-Type: application/json' \
  -d '{"country":"ZMB","new_spending":150,"mode":"both"}'
curl -X POST http://149.102.145.51:8100/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"If Zambia raises health spending to $150, what happens to child mortality?"}'
```

> `/ask` needs a configured LLM for the natural-language step; `/project`, `/scenario`, `/kb`,
> `/build-kb`, `/compile` work without one.

---

# Part I — The System

## 1. System Overview

### 1.1 What Loka is

Loka turns a natural-language question into an answer that is **grounded**, **type-checked**, and
**auditable** — or, when the question falls outside what the system actually knows, into an honest
*"don't know"* rather than a plausible guess.

Concretely, the platform does two things:

- **Build knowledge** — read domain text and produce a formal **ontology** (Ω), together with a
  declaration of what **data** it must store and what **methods** it can apply. This is the
  system's long-term memory / knowledge base (KB).
- **Answer queries** — take a question, formalize it against the ontology, retrieve the relevant
  data or apply the relevant method, and return the result with the assumptions, constraints, and
  an audit trace behind it.

### 1.2 The core claim (why this is not "just an agent")

The center of Loka is the **ontology as a load-bearing formal type system** — not a diagram, not a
database schema. This is the most important idea in this document, and Section 2 is devoted to it.

An ordinary agent can scrape data and compute a number. What it *cannot* do is know the boundary of
its own competence: it answers a question it has no basis for as fluently as one it does. Loka's
ontology is the contract that fixes that boundary. Every query is bound to types declared in Ω; a
target entity or attribute not in Ω is **refused, not guessed**; every method is configured by — and
validated against — Ω, so a method cannot silently reference something the knowledge base does not
model. The working principle is:

> **The LLM proposes; the type system disposes.**

The language model is used only where fluency is needed — reading text into a draft ontology, and
turning a natural-language question into a candidate formal query. Everything load-bearing after
that — validation, grounding, retrieval, method application, the decision record — is deterministic
and reproducible. A hallucinated entity from the model is rejected at the type boundary; it never
reaches an answer.

### 1.3 The two workflows at a glance

```
Workflow A — build the knowledge base
    domain texts ──(LLM proposes)──▶ Ontology Ω + Analysis
                                     + DATA needed / METHODS needed ──▶ KB

Workflow B — answer a query
    question ──(LLM proposes)──▶ formalized query q  ─┬─ asks  → retrieve data from KB
                                                      └─ orders → apply method from KB
             → inform the answer  (or "don't know")  → record it back into the KB
```

Everything in Part I is generic — it does not depend on the demo scenario, which is deferred to
Part II.

---

## 2. The Ontology Ω — the load-bearing type system  ★ core

This is the section the whole platform rests on. If only one thing is read, read this.

### 2.1 What Ω is

An ontology in Loka is a formal object Ω with four primitives:

- **Entity types** (object types) — the kinds of things in the domain, e.g. `Country`. — ✅
- **Attributes** (properties) — typed fields on an entity, inherited along subtyping. — ✅ (type
  + value validation)
- **Relations** (link types) — typed edges between entities, with cardinality. — ✅
- **Actions** — an action verb with a **guard** (precondition) and an **effect**. — ✅ declared ·
  🟡 execution / state write-back not yet wired

Subtyping (⪯) is supported: a subtype inherits and may override its supertype's attributes. Loading
an ontology runs a consistency check (`CΩ`) — a malformed or inconsistent definition is rejected at
load time, not trusted.

### 2.2 What Ω *does* — four jobs, all load-bearing

1. **It grounds and type-checks queries.** A query names target entities/attributes; the grounding
   binder validates them against Ω. An unknown target is a structured error (`UnknownTarget`), not a
   silent bad answer.

2. **It refuses out-of-domain questions.** Because a query must resolve to types in Ω, a question
   that cannot — "will the stock market rise tomorrow?" against a health ontology — returns
   *"don't know"*. The system knows its own limits.

3. **It configures and validates methods.** A method declares the outcome / dial / control fields it
   uses; each must be an attribute declared on the relevant entity in Ω. If Ω does not declare one of
   them, the method is **rejected at startup** — it cannot run against a knowledge base that does not
   model what it needs. *This is tested: removing an attribute from Ω breaks the method.*

4. **It carries governance.** An action's **guard** is a precondition sourced from Ω (e.g.
   `health_exp_per_capita > 0`). The decision stage enforces the guard and records it, so the
   constraint honoured by a decision is the ontology's, not a string hardcoded in application code.

These four are what §9 exercises against real outputs — the refusal (§9.4), the ontology-validated
method (§9.1), and the audit trace tied to Ω's version (§5).

---

## 3. Workflow A — building the knowledge base

**Endpoint:** `POST /build-kb` (✅)

Input is domain text. Output is a validated ontology plus the knowledge it implies, split into what
must be stored and what can be computed:

```
domain texts ──▶ Ontology Ω  +  Analysis (three facets)  +  DATA needed / METHODS needed
```

**The proposer.** When an LLM is configured, it proposes the draft ontology (entities, subtypes,
typed attributes, relations, verbs) — the *texts → LLM → ontology* path. When no LLM is available, a
deterministic keyword builder runs instead, so the workflow is reproducible offline and for
sovereign deployments. Either way, the draft is compiled and run through the ontology loader before
it is trusted — a malformed or inconsistent proposal is rejected, not accepted. The response says
which proposer ran.

**The Analysis — three facets.** The built ontology is decomposed into three facets (following the
agent model's Factual / Cognitive / Communication structure):

- **Factual** — the objective world: entity types, their attributes, relations, factual verbs.
- **Cognitive** — the reasoning content: the methods the agent can apply (this becomes KB.METHODS).
- **Communication** — the communication acts the agent performs: `informs / asks / orders`.

Concrete output, from the deterministic (no-LLM) builder on a short macro-economics text:

```
factual       : [CentralBank, PolicyRate, GDP, CentralBank -sets-> PolicyRate,
                 PolicyRate -affects-> GDP, verb:SETS, verb:AFFECTS]
cognitive     : [method:forecast, method:causal_effect]
communication : [informs, asks, orders]
```

The **DATA needed / METHODS needed** split tells the system what data to acquire for the KB and what
methods to make available — directly feeding Workflow B.

---

## 4. Workflow B — query processing as speech acts

**Endpoint:** `POST /ask` (✅; the NL step needs a configured LLM)

A query in Loka is not a flat parameter bag — it is a **speech act** with a speaker, a listener, a
typed variable, and a predicate. Two forms, dispatched to the two halves of the KB:

```
q = asks(sp, li, ?x:T P(x))              → if P in KB.DATA:    retrieve;  informs(li, sp, P(x)=v)
                                           else                informs(li, sp, "don't know")

q = orders(sp, li, m[in,out] P(x, m(x))) → if m in KB.METHODS: apply m;   informs(li, sp, P(x,m(x)))
                                           else                informs(li, sp, "don't know")

Runtime: every informs(li, sp, P) with a concrete P  →  add P to KB.DATA   (the KB grows as it is queried)
```

The pipeline for one question:

1. **Formalize (LLM proposes).** The model classifies the question into `order` (change a dial,
   apply a method), `ask` (look up a current attribute), or `none`, and extracts the operands.
2. **Ground (type system disposes).** The entity is resolved and the predicate/method is checked
   against Ω and the KB. Anything that does not resolve → *"don't know"*.
3. **Dispatch.** `asks` retrieves from KB.DATA; `orders` applies the method from KB.METHODS.
4. **Inform + record.** The answer is returned as an `informs(...)` act, and the concrete predicate
   is written back into KB.DATA — so an answered query enriches the knowledge base.

**Endpoint:** `GET /kb` (✅) exposes KB.DATA (facts, growing) and KB.METHODS (registered methods),
which makes the "add P to KB" runtime rule directly observable.

Because the query is a *typed* act, the same gate that lets `asks`/`orders` through is the gate that
emits *"don't know"* — refusal and answer come from one mechanism.

---

## 5. Simulation → Policy → Decision memorandum

**Where:** invoked inside `POST /ask` (orders path) and `POST /answer` (the full chain).

An `orders` result is not returned as a bare number — it flows through the decision half of the
system:

```
method result ──▶ Simulation: Scenario Evaluation ──▶ Policy: selection + governance ──▶ Decision memorandum + audit
```

- **Simulation — Scenario Evaluation** 🟡 *basic.* From the method's point estimate and its 95%
  interval, a **nominal / adverse / favourable** scenario triple is derived. This is a labelled
  stand-in for the full simulator (a multi-agent *Agent Society* + a calibrated *EcoFormer*), which
  is a separate ML effort (repo `loka-models`) and is not required to demonstrate the mechanism.

- **Policy — selection + governance** 🟡 *basic.* The nominal scenario is selected under a stated
  **welfare objective** (e.g. minimise the outcome); the ontology's **action guard** is enforced and
  recorded; the projection's **identification label** (see §8.3) is carried through so the read stays
  honest. A full welfare-functional / CVaR *PolicyFormer* is deferred for the same reason.

- **Decision memorandum + audit** ✅ *for what it records.* The memo states the recommendation, the
  welfare objective, the enforced constraint, the identification status, and a **replayable audit
  hash** binding the ontology version + method + exact inputs.

*Receipt.* The same query, changed only in ontology version, produces a different audit hash —
`health-v1 → 0da9d3192b403532`, `health-v2 → 6023f656073dbc01`. A decision is therefore tied to the
exact Ω that authorized it, and re-running the same inputs reproduces the same hash.

---

## 6. Model gateway & behavior-engine port

Loka is **provider-agnostic** about models, by design — sovereign customers must be able to run it
against a self-hosted model with no external calls.

- **Model gateway** ✅ — a single seam (`loka_serving`) resolves an LLM client per purpose
  (ontology-build, grounding, projection). It auto-detects standard `OPENAI_*` / `ANTHROPIC_*`
  environment configuration, so the same code runs against Claude, an OpenAI-compatible proxy, or a
  self-hosted vLLM endpoint. The LLM is only ever the *proposer* (§1.2), so swapping it never changes
  what is load-bearing.

- **Behavior-engine port** 🟡 — a `BehaviorEngine.act(...)` interface is defined for a domain
  **behavior model** (e.g. the research group's Qwen3-32B + LoRA) to plug in as the agent-society
  actor. The interface and a stub implementation exist; the trained model itself is owned by the
  research group and is **not yet plugged in** — the port is the contract it will attach to.

---

## 7. Architecture, API surface, and status

### 7.1 API surface

```
GET  /health              liveness + ontology version                              ✅
POST /build-kb            Workflow A: texts → ontology + facets + DATA/METHODS      ✅
GET  /scenario            show the ontology Ω a method is bound to + validation     ✅
POST /ask                 Workflow B: NL question → speech act → answer/"don't know" ✅ (NL needs LLM)
POST /project             apply the projection method directly (no LLM)             ✅
GET  /kb                  KB.DATA (growing) + KB.METHODS                            ✅
POST /compile             typed query q* → compiled per-question world model W(q,t) ✅
POST /answer              full chain (grounding → W(q,t) → simulate → policy)       🟡 stages 4/5 basic
POST /compile-ontology    compile an externally-authored ontology into W(q,t)       ✅
POST /kb/{id}/ingest      fill a built KB with data rows and causal claims          ✅
```

### 7.2 Implementation status & reasons

**✅ Implemented**
- Ontology Ω: entities, attributes (inherit + validate), relations, subtyping.
- Query grounding + type-check + *"don't know"*.
- Ontology-validated methods (Ω load-bearing) — *tested: dropping an attribute breaks the method*.
- Workflow A: three-facet analysis + DATA/METHODS split.
- Workflow B: `asks`/`orders`/`informs` + add-P-to-KB.
- Causal engine Γ + evidence Kt (platform) — used honestly: the health projection is labelled
  **observational**, not an identified causal effect.
- Production DB backend ports (Postgres / Neo4j) — the demo runs from a real World Bank CSV.

**🟡 Basic / partial — with reason**
- Ontology action *execution* + state write-back — declared (guard + effect); execution planned with
  the action layer.
- Simulation (Agent Society / EcoFormer) — interval-derived scenarios; the full simulator is a
  separate ML effort (`loka-models`).
- Policy (PolicyFormer) — welfare direction + guard + audit; full welfare / CVaR deferred.
- Behavior model — port only; the trained model is owned by the research group and attaches at the
  `BehaviorEngine` port.

---

# Part II — Applied Demo

## 8. Scenario and real data

### 8.1 The scenario

*If a country changes its health spending, what happens to child mortality?* One entity (`Country`),
one outcome (under-5 mortality), one policy dial (health spending per capita), and a set of controls.

### 8.2 The ontology, as generated / authored

The demo runs on the ontology below (`health-v1`) — the same Ω the method is validated against and
that `GET /scenario` reports live:

```yaml
version: health-v1
entities:
  - type: Country
    properties:
      - {name: under5_mortality,      type: double}   # outcome
      - {name: health_exp_per_capita, type: double}   # policy dial
      - {name: gdp_per_capita,        type: double}   # control
      - {name: immunization_dpt,      type: double}   # control
      - {name: sanitation_access,     type: double}   # control
      - {name: water_access,          type: double}   # control
      - {name: fertility_rate,        type: double}   # control
      - {name: urban_pct,             type: double}   # control
verbs:
  - {name: SET_HEALTH_BUDGET, class: institutional}
actions:
  - name: RaiseHealthSpending
    verb: SET_HEALTH_BUDGET
    target: Country
    guard: "health_exp_per_capita > 0"
    effect: "health_exp_per_capita increases to the chosen level"
```

### 8.3 The data (real, not synthetic)

- **Source:** World Bank **World Development Indicators (WDI)** open API — no key required.
- **Panel:** 190 countries × multiple years = **4,382 rows**, 8 indicators (under-5 mortality, health
  spending per capita, GDP per capita, DPT immunization, sanitation, water access, fertility, urban
  %).
- **Why health:** among candidate scenarios, health spending → mortality had a strong, smooth
  signal (fit r² ≈ 0.81 with controls), unlike noisier alternatives that were evaluated and dropped.
- **Method & its honesty label:** a controlled projection (pure-Python OLS) fits `outcome ~ dial +
  controls` across the panel, then projects the target country **anchored to its own current value**,
  holding its controls fixed. The result is labelled **`observational`** — an association-based
  projection, *not* an identified causal effect. This label is carried all the way into the decision
  memo, on purpose.

> The 190 countries are the *training sample* to fit the relationship; the target country (e.g.
> Zambia) is where the fitted relationship is applied. Learning the relationship is statistics (OLS),
> **not** the LLM — the LLM only reads text into the ontology and a question into a formal query.

---

## 9. End-to-end walkthrough (real outputs)

### 9.1 The ontology is bound and validated — `GET /scenario`

```json
{ "entity": "Country",
  "attributes": { "under5_mortality": "double", "health_exp_per_capita": "double", ... },
  "method": { "outcome": "under5_mortality", "dial": "health_exp_per_capita",
              "controls": ["gdp_per_capita", ...], "ontology_validated": true } }
```

The method's fields are the ontology's attributes, and `ontology_validated: true` says Ω authorized
it.

### 9.2 A projection question — `POST /ask` (orders path)

Question: *"If Zambia raises health spending to $150 per capita, what happens to child mortality?"*

```
formalized q : orders(user, loka,
                 project_under5_mortality[Country,health_exp_per_capita->under5_mortality]
                 under5_mortality(ZMB, project_under5_mortality(iso=ZMB, new_spending=150.0)))
answer       : informs(loka, user, under5_mortality(ZMB)=48.551)

Scenario Evaluation :  nominal 48.551 · adverse 82.9 · favorable 14.2
Decision memorandum :  "Raising health_exp_per_capita to 150 reduces under5_mortality for ZMB:
                        49.1 → 48.551"
                       welfare_objective : minimize under5_mortality
                       constraints       : ["health_exp_per_capita > 0"]   (from Ω)
                       identification    : observational
                       audit_manifest    : 0da9d3192b403532
```

(Numbers are from a live run; the wide interval is an honest prediction interval, not hidden.)

### 9.3 A lookup question (IN-domain) — `POST /ask` (asks path)

Question: *"What is Zambia's current child mortality?"* — child mortality **is** an attribute in Ω,
so this is answered with the real value (it is **not** refused):

```
formalized q : asks(user, loka, ?x:Country under5_mortality(x=ZMB))
answer       : informs(loka, user, under5_mortality(ZMB)=49.1)
```

### 9.4 A different, OUT-of-domain question — `POST /ask`

A **separate** example — a question deliberately outside the health ontology (contrast with §9.3,
which is in-domain and returns a number). Question: *"Will Zambia's stock market rise tomorrow?"*

```
answer   : informs(loka, user, "don't know")
```

The health ontology models nothing about stock markets, so **this** query cannot be grounded — and
the system says so, instead of inventing an answer. (This "don't know" applies only to the stock-market
question here, never to the in-domain mortality lookup in §9.3.) **This refusal is the demo's most
important moment:** it is what a grounded, ontology-driven system does and an ordinary agent does not.

### 9.5 The knowledge base grew — `GET /kb`

After the questions above, KB.METHODS lists the projection method and KB.DATA carries the answered
facts, e.g. `under5_mortality(ZMB) = 48.551` — the runtime "add P to KB" rule, visible.
