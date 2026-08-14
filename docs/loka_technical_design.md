# Loka — Technical Design

This document records the choices the platform rests on, what each costs, and what each
guarantee is enforced by. It is not a usage guide; the API surface and a walkthrough are in the
appendices.

Set `LOKA_API` to a running instance to follow the commands (`uvicorn loka_api.app:app` serves
one locally on `:8000`).

---

## §1 Problem and goals

A system that answers questions over an organisation's data fails in a specific way: it answers
everything. Asked something it has no basis for, it produces a fluent, well-formatted, wrong
answer, and nothing in the pipeline distinguishes that from a correct one. For decisions with
consequences, the missing property is not accuracy — it is a **boundary**.

Loka is built for one property: **an answer is produced only when the ontology can ground the
question, the data can support the claim, and the result can be traced back to what authorised
it.** Everything else follows from making that structural rather than best-effort.

A solution must therefore:

1. **Decide** whether a question is answerable, before answering it, on grounds that can be
   inspected.
2. **Refuse with a reason** that identifies which check failed, not a generic apology.
3. Keep **what is measured** separate from **what is computed**, permanently.
4. Never let a claim be **stronger than the interval** behind it.
5. **Distinguish an estimate that cannot be made from one that can be made and explains little.**
   These are opposite failures, and a report carrying only a point estimate makes them look
   identical. §7.4 shows both occurring in real data; this requirement was forced by the cases,
   not designed in advance.
6. Make a decision **replayable** — same authority, same inputs, same data, same answer.

Explicitly **not** goals: to be a data warehouse, to be a general reasoner, to answer without a
modelled domain, or to reach an answer with no human in the loop.

---

## §2 Design decisions

### 2.1 Ω is a type system, not a database schema

The ontology is a contract about what can be asked and answered, validated at load by CΩ and
consulted on every query. It does not describe storage.

*Considered instead:* SQL views over the customer's warehouse; OWL/RDF with a DL reasoner.

*Why:* we need a **decidable boundary of competence** — a question either resolves to declared
types or is refused. OWL is far more expressive, but its open-world semantics treats an unstated
fact as possibly true, which is the behaviour we are eliminating. SQL views give no boundary:
everything is expressible, so nothing is ever refused.

*Cost:* six base types, single inheritance, no axioms, no cardinality *reasoning*. Disjointness
and number restrictions cannot be expressed. The alternative buys expressiveness by giving up the
one property being built for.

### 2.2 The gate is at the query boundary, not in a data pipeline

Data stays where it is. Ω is consulted when a query is formed and when a route is walked; nothing
is rebuilt into ontology-shaped objects.

*Considered instead:* materialising an object layer — ingest and store every entity as the
ontology describes it.

*Why:* a materialised layer must be kept in sync, and each sync failure becomes a confidently
wrong answer. Commercially, sovereign customers do not export data; a design requiring a copy is
unsellable to them.

*Cost — an open gap:* because we do not own storage, the correspondence between an ontology
attribute and a physical column is **unchecked**. R8 verifies a relation's link field is declared
on both types; nothing verifies that `Product.weight_g` maps to a column actually holding grams.
A mis-mapped column yields a well-typed, fully audited, wrong answer. Closing it needs a mapping
declaration and validation at ingest — `validate_values` exists for this and is not yet on the
query path. §7.4 gives an instance of this gap in the live scenario.

### 2.3 The model proposes in two places; both outputs are checked

An LLM appears exactly twice: text → draft ontology, and question → candidate formal query.
Everything after is deterministic.

*Considered instead:* free-form text-to-SQL; the model answering directly over retrieved context.

*Why:* both make the model's output load-bearing with no downstream check that can catch a
hallucinated table or a plausible wrong figure. Here the output is a proposal over a vocabulary
**generated from Ω**, and that same vocabulary is the acceptance test:

```python
omega_attrs = sorted(engine.properties_of(ENTITY))        # the prompt's vocabulary
proposal    = formalize_query(question, llm, model, attributes=omega_attrs, entity=ENTITY)
if not isinstance(attribute, str) or attribute not in omega_attrs:   # ...and the gate
    return _dont_know(..., "not_in_ontology")
```

Generating both from one source is the point: swap the ontology and prompt and gate move
together. Where SQL is generated, it is assembled from validated identifiers with bound
parameters, never written by the model (§5.3).

*Cost:* an ontology that omits a concept makes it unaskable rather than answerable-with-a-caveat.
The burden moves onto ontology quality — hence 2.4.

### 2.4 Human review is a state in a machine, not a documented process

A generated ontology is a `draft`; it becomes `validated` by passing CΩ and `published` only when
a person approves it. `/answer` accepts published only, else 409.

*Considered instead:* trusting CΩ, with review as a step teams are expected to follow.

*Why:* CΩ catches structural faults but not a proposal that is well-formed and wrong. The question
a reviewer asks is *"how do you know the generated ontology is correct?"*, and the only defensible
answer is architectural: **the model need not be correct, because its output cannot reach a
decision without passing CΩ and a person.** A documented process gives no such guarantee; a 409
does.

*Cost:* nothing is answerable straight out of `/build-kb`. The fully automatic text-to-answer path
does not exist and will not.

### 2.5 A relation declares the field it is traversed by

Every relation carries `via`; CΩ R8 requires it on both endpoints.

*Considered instead:* a naming convention — to reach `T`, use column `{t}_id`.

*Why:* otherwise the route is derived from Ω while traversal stays a hidden convention in code —
the same defect as hardcoding attribute names in a prompt. With `via`, route and traversal come
from one source, and a relation omitting it is reported as not traversable rather than guessed at.

*Cost:* Ω contains physical field names, so it is not purely conceptual. Renaming a column
requires a new ontology version — which, given 2.4's freeze-on-publish, is the correct
consequence.

---

## §3 Architecture

### 3.1 Packages

| Package | Responsibility | Modules / lines |
|---|---|---|
| `libs/loka-schemas` | shared contracts **and the protocols the services are written against** | contracts |
| `services/ontology` | Ω, CΩ (R1–R8), engine, route search, traversal, builders | 7 / 1523 |
| `services/grounding` | NL → q*: proposer (LLM or keyword) + deterministic binder | 4 / 295 |
| `services/compiler` | binds Ω + state + mission + q* into W(q,t) | 1 / 87 |
| `services/state` | world state Eₜ, snapshot hashing | 1 / 71 |
| `services/causal` | causal graph Γ, admissibility, slices | 4 / 370 |
| `services/knowledge` | evidence base Kt, meta-synthesis, contradiction records | 1 / 167 |
| `services/adapters` | read-only access: in-memory, Postgres, World Bank; safe SELECT planner | 4 / 395 |
| `services/serving` | model gateway, behavior-engine port, audit of model resolutions | 3 / 286 |
| `services/api` | HTTP surface and orchestration | 16 / 2772 |

### 3.2 Dependency inversion is the architecture

Not a layered stack. **No service package imports another service package.** Each is written
against protocols declared in `loka-schemas` (`OntologyView`, `StateView`, `StateStore`,
`CausalSlicer`, …), and assembly happens in exactly one place.

```
                    loka-schemas         contracts + protocols; depends on nothing
                         ▲
   ┌──────────┬──────────┼──────────┬──────────┬──────────┬──────────┐
ontology    state     causal    knowledge  adapters   grounding   compiler
                                                                    serving   (imports nothing)
   └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                         ▲
                        api            the only package that composes the others
```

The compiler is the clearest case. It computes W(q,t) from an ontology, a state and a causal
slicer, and imports none of them:

```python
# compiler.py — the entire external import
from loka_schemas import (OntologyView, StateView, CausalSlicer, ...)

def compile_wqt(engine: OntologyView, state: StateView, mission, query, ...)
```

Two consequences follow, and both are properties rather than conveniences:

- **Every package is testable against a stub.** The binder is tested with a stub ontology not
  because grounding is special, but because no package can reach a concrete implementation of
  anything.
- **Replacing an implementation cannot ripple.** A different ontology engine, state store or
  causal backend satisfies the same protocol; nothing else recompiles or changes.

The read and write surfaces are separate protocols for the same reason: the compiler is handed a
`StateView` and therefore cannot mutate the state it is compiling against, while an ingestion path
asks for `StateStore`. A concrete store satisfies both.

The cost is that a reader cannot see, from a package, which implementation it will run against —
that is only visible at the composition site in `api`.

### 3.3 Three seams

1. **Model gateway** (`serving`). Every model request goes through `llm_for(purpose)` /
   `model_for(purpose)`; the provider is selected by environment. Claude, an OpenAI-compatible
   proxy and a self-hosted vLLM endpoint are interchangeable, and every resolution is logged so
   "which model answered what" is recoverable. Reasoning-model token accounting is absorbed here —
   the adapter floors `max_tokens` and raises a named error when a response carries no content,
   rather than letting each call site discover it as a parse failure.

2. **Data adapters** (`adapters`). All reads implement one contract (`authenticate` → `query` →
   typed rows with lineage). Postgres reads through a read-only connection with a server-side
   cursor. Swapping the store does not reach the ontology or the compiler.

3. **Behavior-engine port** (`serving.behavior`). `BehaviorEngine.act(...)` is the attachment
   point for a domain behavior model. A deterministic stub ships so the surrounding machinery
   runs; the trained model is owned by the research group and is not attached.

### 3.4 The path of one query

```
question
  │
  ├─ serving.llm_for(...)                model proposes; vocabulary generated from Ω
  ├─ ontology.properties_of(entity)      gate: predicate declared? → else refuse (typed)
  ├─ speechact                           q = asks(...) | orders(...)
  │     ├─ asks   → KB.DATA              actual world only, with provenance
  │     └─ orders → KB.METHODS → projection    effect + its CI (not the level interval)
  ├─ policy                              scenarios; welfare direction; Ω's guard;
  │                                      claim withheld if the CI spans zero;
  │                                      qualified if the fit explains little
  └─ audit hash                          Ω version + method + inputs + controls
                                         + digest of the fitted sample
```

For a relational query the middle differs: `path_between` derives the route from R, `follow` walks
it by each relation's `via`, and a target reached only through its supertype is reported as
requiring narrowing rather than returned as certain.

---

## §4 Formal model

### 4.1 Ω

Ω = (E, A, R, ⪯, C, Actions), authored in YAML.

- **E** entity types; **A** typed attributes, inherited along ⪯, a subtype may override
- **R** directed relations with a cardinality and a link field `via`
- **⪯** single inheritance, a partial order (R6)
- **C** typing constraints: which entity types a verb may act on
- **Actions** an action verb with a **guard** (precondition) and an **effect**

### 4.2 CΩ — eight rules, checked at load

| Rule | Rejects |
|---|---|
| R1 | duplicate property in one entity |
| R2 | `subtype_of` naming an undeclared type |
| R3 | a relation endpoint that is not declared |
| R4 | a constraint over an undeclared verb or type |
| R5 | an action over an undeclared verb or target |
| R6 | a cycle in ⪯ |
| **R7** | a subtype override that breaks substitutability |
| **R8** | a link field absent on either end of a relation |

A rejection names the rule that failed, so an author is told what to fix:

```python
load_ontology_str("version: t\nentities:\n  - type: A\n    subtype_of: Ghost\n")
# OntologyLoadError: entity A has subtype_of=Ghost, which is not defined
```

R7 is what keeps ⪯ sound. A subtype may repeat a property's type or **narrow** it
(`integer ⊑ double`, `date ⊑ timestamp`); widening it, changing it to an unrelated type, or
relaxing `required` to optional is refused, because a value of the subtype would then not be
usable where the supertype is expected. It is checked against every ancestor, not only the
immediate parent.

```python
# loader.py — _check_override_compatibility
allowed = _NARROWABLE_TO.get(p.base_type, frozenset())
if sub_prop.base_type != p.base_type and sub_prop.base_type not in allowed:
    raise OntologyLoadError(
        f"entity {ent.name} overrides inherited property {p.name} with type "
        f"{sub_prop.base_type} but {parent.name} declares it as {p.base_type}; "
        f"a subtype may only repeat the type or narrow it")
```
> `test_override_widening_the_type_is_rejected`,
> `test_override_is_checked_against_every_ancestor_not_just_the_parent`

R8 exists because §5 derives routes from R: the field each relation is walked by must exist on
both types, or Ω promises a path it cannot walk.
> `test_c_omega_rejects_a_via_field_missing_on_either_side`

The full CΩ in the design notes is roughly 250 rules; these eight are the structural core, and a
Soufflé/Datalog path exists (`souffle_checker.py`) for the semantic remainder. What is
unimplemented is stated in §8, not implied to exist.

### 4.3 q — a query is a speech act

A query is not a parameter bag. It carries a speaker, a listener, a typed variable and a
predicate, and dispatches to one of the two halves of the KB:

```
q = asks(sp, li, ?x:T P(x))              → P ∈ KB.DATA    ? retrieve : informs(li,sp,"don't know")
q = orders(sp, li, m[in,out] P(x,m(x)))  → m ∈ KB.METHODS ? apply    : informs(li,sp,"don't know")
```

The value is not the notation: because the query is a *typed* act, the gate that admits
`asks`/`orders` is the same gate that emits the refusal, so answer and refusal come from one
mechanism rather than two code paths that can disagree.

### 4.4 W(q,t) — the per-question world model

`compile_wqt(Ω, Eₜ, mission, q)` freezes everything an answer depends on into one object: the
state slice for the query's targets, the causal slice Γ(q) when present, the mission's welfare
terms and hard constraints, and a manifest pinning the ontology version and a state snapshot
hash. Compilation is deterministic: the same inputs produce the same W(q,t), which is what makes
§6's replay guarantee possible.

---

## §5 Relations, traversal, and generated SQL

### 5.1 Route derivation

Route search is breadth-first over R, in both directions, respecting ⪯ at every step:

```python
# engine.py — path_between
steps = [(r, True,  r.to_type)   for r in self.relations_from(current)] + \
        [(r, False, r.from_type) for r in self.relations_to(current)]
```

Walking uses only the declared field — the same field in either direction, which is why a relation
reverses for free and why "which seller shipped this order" and "which orders does this seller
appear in" are one mechanism:

```python
# traverse.py — follow
nxt  = rel.to_type if forward else rel.from_type
keys = {r[rel.via] for r in rows if r.get(rel.via) is not None}
rows = [r for r in rows_of_type(engine, dataset, nxt) if r.get(rel.via) in keys]
```

**Reaching a subtype is not the same as reaching a supertype.** Landing on a subtype of the target
always reaches it — every `BulkyProduct` is a `Product`. Landing on a supertype does not: an
`Order` reaches `Product`, but whether that product is bulky is a runtime narrowing the type system
cannot guarantee. Such a path is withheld unless the caller asks for it, and `needs_narrowing`
distinguishes it from "no route at all". A bare "unreachable" would have been misleading, and it is
the distinction a reader checking soundness will look for.

> `test_a_multi_hop_route_is_derived_from_the_declared_relations`,
> `test_relations_are_walked_backwards_for_impact_range`,
> `test_reaching_a_subtype_requires_narrowing_and_is_reported_as_such`,
> `test_a_relation_without_via_is_not_traversable`

### 5.2 Eligibility and consequence

An action's guard is the rule; the entity it applies to and the attribute it names both come
from Ω:

```python
# supply.py — impact_of_tightening
action = next((a for a in engine.action_types() if a.name == action_name), None)
attribute, op, old_threshold = parse_guard(action.guard)
if attribute not in engine.properties_of(action.target):
    return {"error": f"guard references '{attribute}', which {action.target} does not declare"}
```

Rows that satisfied the old threshold and not the new one lose eligibility; the consequence is
then followed along R. Nothing in the service knows that products have sellers.

### 5.3 Generating SQL

The model never writes SQL. It selects an entity and attributes; the statement is assembled from
identifiers that have been validated, with every value bound as a parameter:

```python
# sql_planner.py
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_check_ident(table, "table")
cols = ", ".join(_check_ident(c, "column") for c in columns)
clauses.append(f"{_check_ident(key, 'filter column')} = %s")
params.append(value)                       # values never enter the statement text
```

Injection is impossible by construction rather than filtered for: an identifier that is not a
plain identifier is refused, and a value never reaches the statement.
> `test_injection_attempt_rejected_in_table`, `test_injection_attempt_rejected_in_column`

**Status.** 🟡 The planner is implemented and tested but has no caller yet. Reads that run today
go through `PostgresAdapter`, which builds its own statement with psycopg's `sql.Identifier`
quoting and bound parameters over a read-only connection — safe, but `SELECT *` rather than a
projection of the attributes Ω declares. The end-to-end chain (question → q* → generated SQL →
live database) is not connected; the shipped scenarios read CSV. The entity-to-table mapping is a
configuration dict with no validation, which is the concrete form of the gap named in §2.2.

---

## §6 Guarantees, and what is not guaranteed

Each guarantee holds because of a specific mechanism, not because of care. The mechanism is shown.

**S1 — A query that cannot be grounded in Ω is refused, not answered.**
The vocabulary offered to the model and the acceptance gate come from the same call (code in
§2.3). There is no path that reaches an answer with an undeclared predicate, and the refusal names
which check failed (`unknown_entity` · `not_in_ontology` · `no_data` · `unformalizable`). The
decisive test deletes an attribute from Ω and shows the service refuses to start rather than
answering with a method the ontology does not authorise.
> `test_ask_refuses_a_predicate_not_declared_in_omega`,
> `test_removing_an_attribute_from_omega_breaks_the_lookup`

**S2 — A computed value is never returned as an observation.**
KB.DATA is keyed by `(entity, predicate, scenario)` and every fact carries provenance. The
invariant is enforced at the write, not by discipline at the read:

```python
# speechact.py
if prov.kind == "derived" and scenario_id is ACTUAL:
    raise ValueError("a derived fact may not be written into the actual world; "
                     "pass a scenario_id identifying the counterfactual it holds in")
```
> `test_projection_never_overwrites_an_observation` (unit and over HTTP),
> `test_derived_fact_cannot_be_written_into_the_actual_world`

**S3 — A projection's conclusion is never stronger than the evidence behind it.**
Two distinct checks, because there are two ways to be weaker than a conclusion. First, the
interval: because the projection is anchored, `point = y_cur + β_dial·Δt`, so the relevant
uncertainty is the dial coefficient's, not residual spread across the panel:

```python
# projection.py — the variance that matters is the dial coefficient's
var_beta_dial = s2 * _solve(XtX, e_dial)[1]        # e_dial selects column 1: the dial
se_effect     = math.sqrt(max(var_beta_dial, 0.0)) * abs(delta_t)

# policy — a direction is claimed only if the interval excludes it
significant = eff_lo is not None and (eff_lo > 0 or eff_hi < 0)
```

Second, explanatory power. A determined direction over a model that accounts for almost nothing is
a different situation from an undetermined one, and must not read the same (requirement §1.5):

```python
WEAK_FIT_R2 = 0.10           # the threshold is published, not applied silently
"explains_little": isinstance(r2, (int, float)) and r2 < WEAK_FIT_R2
```

When it fires, the recommendation says so in words rather than leaving it to a field a reader may
not open: *"The direction is determined, but the fitted model accounts for 1.3% of the variation …
precise about a relationship that explains little of any individual case."*

> `test_memo_refuses_to_claim_a_direction_when_the_effect_straddles_zero`,
> `test_a_determined_direction_over_a_weak_fit_reads_differently`,
> `test_the_weak_fit_threshold_is_published_not_applied_silently`

This guarantee covers the projection path. `asks` retrieves an observation and has no interval;
its honesty guarantee is S2's provenance, not this one.

**S4 — Only a published ontology authorises an answer.**

```python
# app.py — /answer
if rec is not None and rec.state != "published":
    raise HTTPException(status_code=409, detail=(
        f"ontology {oid} is '{rec.state}', not 'published': a generated ontology must "
        f"pass CΩ and be approved before it can authorize an answer. "
        f"{len(rec.review)} review item(s) outstanding"))
```
Publishing freezes the ontology; editing means publishing a new version.
> `test_a_draft_cannot_authorize_an_answer`, `test_a_published_ontology_is_frozen`

**S5 — A decision is replayable.**
The audit hash binds the ontology version, the method, the inputs, the control values held fixed,
and a digest of the sample actually fitted. The preimage is published alongside the hash, so an
auditor recomputes rather than trusts:

```python
# the exact string hashed
return (f"{inputs['ontology_version']}|{inputs['method']}|{inputs['entity']}|"
        f"{inputs['outcome']}|{inputs['dial']}|{inputs['dial_change']}|"
        f"{inputs['outcome_current']}|{controls}|"
        f"{inputs['sample_digest']}|{inputs['sample_n']}|{inputs['sample_params']}")
```
A data revision changes the hash; the same inputs and data reproduce it. Where the digest is
absent the memo says `replayable: false` rather than hashing a null and looking sound.
> `test_audit_hash_binds_the_fitted_sample`,
> `test_audit_inputs_are_published_so_the_hash_can_be_recomputed`

### Not guaranteed

- **That an ontology attribute maps to the right physical column** (§2.2). The type system checks
  names against Ω, not the meaning of a column. This is the largest open gap; §7.4 gives a live
  instance.
- **That a published ontology is semantically correct** — only that CΩ passed and a person
  approved it. The guarantee is about authority, not truth.
- **Causal identification.** Projections are labelled `observational` and carry that label into
  the memo. Appendix D shows a case where the effect is not identifiable at all.
- **That the model proposes correctly** — only that an incorrect proposal is caught before it
  reaches an answer.
- **Freshness.** Nothing detects that the underlying data has moved; the audit digest detects it
  *after the fact*, by producing a different hash.

---

## §7 Case study — the Olist marketplace

One scenario is carried end to end, chosen because it exercises all six parts of Ω. A second
ontology (`health-v1`, World Bank WDI) is also loaded; it declares one entity and no relations, so
R, ⪯ and C sit unused there. That contrast is the reason for two ontologies rather than one:
**a single-table domain uses less than half the type system.** The typing gate and refusal
machinery of §6 can be verified there — S1 and S3 cite its tests — while the relational machinery
cannot. Statistical notes on the health projection are in Appendix D.

### 7.1 Where the data comes from

**Source.** The Olist Brazilian e-commerce dataset, published by Olist itself at
`github.com/olist/work-at-olist-data`. Real marketplace records, anonymised by the publisher —
store and partner names were replaced before release. No licence request, no API key.

**What is in this repository, and what is not.**

| | Location |
|---|---|
| The ontology | `examples/supply_ontology.yaml` — versioned with the code |
| The build script | `examples/build_supply_data.py` — the only per-scenario code |
| The source tables | not committed (~44 MB); `--download` fetches them |
| The built entity tables | not committed; produced by the script |

```bash
python examples/build_supply_data.py --download
LOKA_SUPPLY_DATA=examples/supply_data uvicorn loka_api.app:app
```

Two commands reproduce every number in this section. The source tables are excluded deliberately:
committing them would make the repository the system of record for data it does not own, and the
audit digest (S5) is what establishes which snapshot a result came from — not a file in git. The
supply tests skip, rather than pass, when the data is absent.

**What the script produces.** One CSV per entity type declared in Ω, keyed so each relation can be
walked by its declared `via` field:

```
Seller          3,095      seller_id, seller_state, on_time_rate
Product        31,060      product_id, weight_g, volume_cm3, category
BulkyProduct    1,891      the subset above the ShipStandard limit
OrderItem     112,650      item_id, order_id, product_id, seller_id, freight_value, price
Order          99,441      order_id, customer_id, days_late, status
Customer       99,441      customer_id, customer_state
                ───────
              347,578 rows
```

**Nothing is pre-joined, and this is the load-bearing choice.** A single wide table would already
contain each order's seller, so the multi-hop questions of §5 would be answered by the CSV rather
than by Ω, and the route derivation would be untested. Keeping the tables separate is what forces
the system to consult the ontology.

Two columns are computed rather than read, and both are stated in the ontology's own field
descriptions so a reader is never guessing which values are measured:

```
Product.volume_cm3    length × height × width
Seller.on_time_rate   share of that seller's delivered lines that arrived by the promised date
```

The bulky threshold is not a constant in the script — it is read from the `ShipStandard` guard, so
the subtype boundary and the eligibility rule remain one rule.

### 7.2 The ontology

Authored by hand, reviewed, and published (§2.4). 87 lines; the parts that matter:

```yaml
version: supply-v2

entities:
  - type: Product
    properties:
      - {name: product_id, type: string, required: true}
      - {name: weight_g,   type: double, description: "shipping weight in grams"}
      - {name: volume_cm3, type: double, description: "length x height x width, cm³"}

  - type: BulkyProduct
    subtype_of: Product                      # ⪯ — decides what is permitted
    properties:
      - {name: weight_g, type: double, description: "above the standard-service limit"}

  - type: OrderItem                          # the junction, modelled as a real entity
    properties:
      - {name: item_id,    type: string, required: true}
      - {name: order_id,   type: string}
      - {name: product_id, type: string}
      - {name: seller_id,  type: string}

relations:
  - {name: contains,     from: Order,     to: OrderItem, via: order_id,    cardinality: one_to_many}
  - {name: of_product,   from: OrderItem, to: Product,   via: product_id,  cardinality: many_to_one}
  - {name: fulfilled_by, from: OrderItem, to: Seller,    via: seller_id,   cardinality: many_to_one}
  - {name: placed_by,    from: Order,     to: Customer,  via: customer_id, cardinality: many_to_one}

verbs:
  - {name: SHIP_STANDARD,  class: factual}
  - {name: NOTIFY_DELAY,   class: communicative}
  - {name: SUSPEND_SELLER, class: institutional}

actions:
  - name: ShipStandard
    verb: SHIP_STANDARD
    target: Product
    guard: "weight_g <= 10000"               # eligibility lives here, not in code
```

What each part is doing, and why it is not decoration:

| Part of Ω | Declared as | Why it carries weight |
|---|---|---|
| **E** | 6 entity types | — |
| **A** | typed attributes, inherited along ⪯ | `BulkyProduct` inherits `product_id`, overrides `weight_g` — R7 checks the override |
| **R** | 4 relations, each with `via` | §5 derives routes from these; without `via`, R8 rejects the ontology |
| **⪯** | `BulkyProduct ⪯ Product` | decides whether `ShipStandard` applies at all |
| **C** | 2 typing constraints | which entity types each verb may act on |
| **Actions** | `ShipStandard` guard | the eligibility rule; the service holds no copy of it |

**`OrderItem` is a real entity, not a convenience.** About 10% of orders hold several items, and
about 4% of products are sold by more than one seller, so the seller belongs to the order line
rather than to the product. Modelling the junction keeps the declared cardinalities true of the
data — and makes `Order → Seller` a genuine two-hop route rather than a direct edge.

### 7.3 What was verified

Five things, each end to end against the live data.

**① A route is derived from Ω, not written by hand.**
```
GET /supply/route?from_type=Order&to_type=Seller
→ hops 2, route ["contains>(via order_id)", "fulfilled_by>(via seller_id)"]
```

**② Reaching a supertype is distinguished from reaching the target.**
```
GET /supply/route?from_type=Order&to_type=BulkyProduct
→ requires_narrowing: true
```
An `Order` reaches `Product`; whether that product is bulky is a runtime check the type system
cannot make. Reported as such, rather than returned as certain or as "no route".

**③ A type that does not exist is refused, not approximated.**
```
GET /supply/route?from_type=Customer&to_type=Warehouse
→ 404  'Warehouse' is not an entity in ontology supply-v2
```

**④ Changing one rule in the ontology, and following the consequence along R.**
```
POST /supply/impact  {"action":"ShipStandard","new_threshold":5000}

guard      weight_g <= 10000  →  5000
products   2,262 newly ineligible
  8,596 order lines   of_product<
  7,810 orders        of_product< → contains<
    726 sellers       of_product< → fulfilled_by>
  7,810 customers     of_product< → contains< → placed_by>
```
The rule, the entity it applies to, and the attribute it names all come from Ω. Products already
over the old limit are not counted as *newly* ineligible. Nothing in the service knows that
products have sellers — that reach is followed along the declared relations, backwards where
needed, which is why the same mechanism answers "which orders does this seller appear in".

**⑤ The build → review → publish → answer path.** A generated ontology enters as a draft with a
review checklist; `POST /answer` against it returns 409; a reviewed edit passes CΩ to `validated`;
approval makes it `published` and frozen; the same question then answers. Appendix B has the
commands.

### 7.4 What the cases exposed, and what this one does not establish

**The requirement in §1.5 came from here.** Two projections, two opposite failures:

| | health-v1, spending → mortality | supply, weight → lateness |
|---|---|---|
| effect | −0.549 | +0.388 |
| 95% CI | [−1.329, 0.230] — spans zero | [0.208, 0.568] — excludes zero |
| r² | 0.809 | **0.013** |
| reading | cannot be estimated | estimated precisely, explains ~1% |

The first is *"we cannot tell"*; the second is *"we can tell, and it accounts for almost nothing
for any individual order"*. A report carrying only a point estimate makes them identical, which is
why the memo now states both conditions in words (S3).

**What this case does not establish:**

- `days_late` is measured against the marketplace's **own** promised date. A seller can appear
  punctual because its promise was generous. No amount of cleaning fixes this; it is a validity
  limit of the measure, and it is stated in the ontology's field description rather than only here.
- This is marketplace data, not an industrial supply chain. There is no bill of materials and no
  multi-tier supplier structure, so a spec change propagates one product deep, not through an
  assembly.
- `Seller.on_time_rate` is **derived by the build script**, not reported by the source. The
  `SuspendSeller` guard therefore tests a computed quantity — legitimate, but the ontology does not
  record that the attribute is derived. **This is an instance of the gap named in §2.2.**
- No causal claim is made anywhere in this scenario. The impact analysis is a deterministic
  consequence of a rule change over the current data, not a prediction of what would happen.

---

## §8 Evolution

**Implemented.** Ω with R1–R8; routes derived from Ω and traversed by declared link fields;
guard-driven eligibility and consequence; four typed refusals; provenance-separated KB; ontology
lifecycle with an automatic review checklist; replayable audit hash; provider-agnostic model
gateway. 230 tests; `mypy --strict` and `ruff` clean.

**Basic, and labelled as such in the output.** Scenario evaluation derives a
nominal/adverse/favourable triple from the effect interval and marks its probabilities
`prob_basis: placeholder` — they are a nominal/bounds split, not a calibrated distribution. The
policy stage applies a welfare *direction*, the ontology's guard and the audit hash, not a welfare
functional. A calibrated simulator and a policy model are a separate ML effort.

**Not built, and what each needs.**

| Missing | What it needs |
|---|---|
| Column-to-attribute mapping validation (§2.2) | a mapping declaration in Ω or alongside it, plus `validate_values` called at ingest |
| End-to-end generated SQL against a live database (§5.3) | the planner wired into the read path, and the entity-to-table mapping validated |
| A `derived` marker on attributes (§7.4) | one field in `Property`; the checklist would then flag a guard over a derived quantity |
| Action execution and state write-back | guards are declared and evaluated; an executor and a governed write path |
| Planning | goal decomposition; currently no substitute is claimed |
| Semantic CΩ beyond R1–R8 | the Soufflé path exists; the rule set has to be written |
| The multi-agent simulation | the behavior-engine port exists; the trained model attaches there |
| Service separation | the dependency inversion of §3.2 already permits it; a shared store for `KB` and `OntologyStore` is the prerequisite |

---

## Appendix A — API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + ontology version |
| `GET /scenario` | the health method and the Ω attributes it is bound to |
| `POST /project` | apply the projection method directly (no model call) |
| `POST /ask` | NL question → speech act → answer, or a typed refusal |
| `GET /kb` | KB.DATA (actual world) and `all_facts` (incl. counterfactuals) |
| `POST /build-kb` | text → draft ontology + review checklist |
| `GET /ontology`, `GET\|PUT /ontology/{id}`, `POST /ontology/{id}/publish` | review lifecycle |
| `GET /supply/scenario` | entities, relations with link fields, guarded actions, row counts |
| `GET /supply/route?from_type=&to_type=` | the route Ω declares between two types |
| `POST /supply/impact` | tighten a guard → what loses eligibility, what it reaches |
| `POST /compile` | typed query q* → W(q,t) |
| `POST /answer` | the full chain (grounding → W(q,t) → simulate → policy) |
| `POST /compile-ontology` | compile an externally-authored ontology into W(q,t) |
| `POST /kb/{id}/ingest` | fill a built KB with data rows and causal claims |

## Appendix B — Walkthrough

```bash
B="${LOKA_API:-http://localhost:8000}"

# routes are derived, not hand-written
curl -s "$B/supply/route?from_type=Order&to_type=Seller"
curl -s "$B/supply/route?from_type=Order&to_type=BulkyProduct"   # requires_narrowing: true
curl -s "$B/supply/route?from_type=Customer&to_type=Warehouse"   # 404, not an entity

# a rule change and its consequence
curl -s -X POST "$B/supply/impact" -H 'Content-Type: application/json' \
     -d '{"action":"ShipStandard","new_threshold":5000}'

# refusal, with the check that failed
curl -s -X POST "$B/ask" -H 'Content-Type: application/json' \
     -d '{"question":"Will Zambia'"'"'s stock market rise tomorrow?"}'

# a draft cannot authorise an answer
curl -s -X POST "$B/build-kb" -H 'Content-Type: application/json' \
     -d '{"texts":["The Central Bank sets the Policy Rate, which affects GDP."]}'
# → state draft + review checklist; POST /answer with its kb_id → 409
```

## Appendix C — Claim → test index

| Claim | Test |
|---|---|
| Ω refuses an undeclared predicate | `test_ask_refuses_a_predicate_not_declared_in_omega` |
| Removing an attribute from Ω stops the service starting | `test_removing_an_attribute_from_omega_breaks_the_lookup` |
| A projection never overwrites an observation | `test_projection_never_overwrites_an_observation` |
| A derived fact cannot enter the actual world | `test_derived_fact_cannot_be_written_into_the_actual_world` |
| No direction claimed when the CI spans zero | `test_memo_refuses_to_claim_a_direction_when_the_effect_straddles_zero` |
| A determined direction over a weak fit reads differently | `test_a_determined_direction_over_a_weak_fit_reads_differently` |
| The weak-fit threshold is published, not silent | `test_the_weak_fit_threshold_is_published_not_applied_silently` |
| Scenarios use the effect CI, not the level PI | `test_scenarios_use_the_effect_interval_not_the_level_interval` |
| A signed outcome is not floored by default | `test_a_signed_outcome_is_not_floored_by_default` |
| A draft cannot authorise an answer | `test_a_draft_cannot_authorize_an_answer` |
| A published ontology is frozen | `test_a_published_ontology_is_frozen` |
| The audit hash binds the fitted sample | `test_audit_hash_binds_the_fitted_sample` |
| The audit hash can be recomputed independently | `test_audit_inputs_are_published_so_the_hash_can_be_recomputed` |
| A subtype override may not break substitutability (R7) | `test_override_widening_the_type_is_rejected` |
| R7 is checked against every ancestor | `test_override_is_checked_against_every_ancestor_not_just_the_parent` |
| A link field must exist on both ends (R8) | `test_c_omega_rejects_a_via_field_missing_on_either_side` |
| Routes are derived from declared relations | `test_a_multi_hop_route_is_derived_from_the_declared_relations` |
| Relations reverse for impact range | `test_relations_are_walked_backwards_for_impact_range` |
| Narrowing is reported, not assumed | `test_reaching_a_subtype_requires_narrowing_and_is_reported_as_such` |
| A relation without `via` is not traversable | `test_a_relation_without_via_is_not_traversable` |
| Generated SQL refuses a non-identifier | `test_injection_attempt_rejected_in_table` |
| A method's attributes must exist in Ω | `test_method_spec_rejects_attribute_not_in_ontology` |

## Appendix D — Statistical notes on the health projection

**The effect is not identifiable in this panel.** `corr(log spending, log GDP) = 0.968` over 4,382
rows; regressing the dial on all controls gives R² = 0.9396, so VIF ≈ 16.5. Dropping GDP leaves r²
**unchanged at 0.809** while |t| moves from 1.38 to 8.02. Fit does not improve; only the standard
error collapses, because two near-collinear regressors were sharing variance. The honest statement
is not "spending reduces mortality" but **"in this panel the effect of spending cannot be separated
from the effect of income"**. Reporting only the specification without GDP would be specification
shopping, so all specifications are reported together:

```
controls                    effect     t        r²
all six (as shipped)        −0.549    −1.38    0.809
without gdp_per_capita      −1.491    −8.02    0.809
immunisation + fertility    −3.229   −20.71    0.780
none                       −10.940   −72.74    0.547
```

**Significance cannot be tuned by choosing a country or an amount.** Because the projection is
anchored, `effect = β·Δt` and `se = √Var(β̂)·|Δt|`, so `t = β/√Var(β̂)` — Δt cancels exactly.

```
ZMB new=100  effect −0.198   t −1.3808
ZMB new=600  effect −1.750   t −1.3812
NGA new=150  effect −0.700   t −1.3804
IND new=150  effect −0.495   t −1.3808
```

|t| is 1.38 regardless. An entire class of quiet parameter-shopping is excluded arithmetically
rather than by policy.
