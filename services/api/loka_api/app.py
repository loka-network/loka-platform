"""Loka Platform HTTP API (FastAPI).

Minimal service surface over the foundation:
  - GET  /health   → liveness + ontology version
  - POST /compile  → a typed query q* → the compiled Scenario World Model W(q, t) as JSON

The natural-language front end (NL → q*) is in loka_grounding; /compile takes a typed query.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from loka_compiler import CompileError, compile_wqt
from loka_schemas import TypedQuery
from pydantic import BaseModel

from .world import World, build_world_from_env


class CompileRequest(BaseModel):
    """A typed query q* posted to /compile."""

    query_id: str
    task_type: str
    targets: list[str]
    signature: str | None = None


class BuildKBRequest(BaseModel):
    """Domain texts posted to /build-kb (Workflow A: texts -> ontology + DATA/METHODS)."""

    texts: list[str]
    #: Override the extraction instruction. The default covers entities, their attributes,
    #: relations, verbs and the DATA/METHODS needs — not cardinality, guards or norms, which a
    #: reviewer supplies. A domain that needs a different emphasis passes its own rather than
    #: having one opinion compiled into the platform; whichever ran is recorded on the draft.
    system_prompt: str | None = None


class IngestRequest(BaseModel):
    """Data rows and causal claims to fill a built KB's DATA/METHODS needs."""

    data: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []


class ProjectRequest(BaseModel):
    """A projection query (Workflow B, orders -> KB.METHODS): move a policy dial, project outcome.

    The health scenario: for ``country`` (ISO3), if health spending per capita -> ``new_spending``,
    project under-5 mortality. ``mode`` = controlled | naive | both.
    """

    country: str
    new_spending: float
    mode: str = "both"


class AskRequest(BaseModel):
    """A natural-language projection question (Workflow B, full NL): the LLM extracts the params."""

    question: str
    mode: str = "both"


# health scenario config (which panel columns are outcome / dial / controls)
_H_OUTCOME = "under5_mortality"
_H_DIAL = "health_exp_per_capita"
_H_CONTROLS = [
    "gdp_per_capita", "immunization_dpt", "sanitation_access",
    "water_access", "fertility_rate", "urban_pct",
]
_H_LOG = ["health_exp_per_capita", "gdp_per_capita"]


def _load_health_panel() -> list[dict[str, Any]] | None:
    """Load the real World Bank health panel (env override, else repo/cwd examples/)."""
    import csv
    import os

    here = os.path.dirname(__file__)
    candidates = [
        os.getenv("LOKA_HEALTH_PANEL"),
        os.path.join(here, "..", "..", "..", "examples", "health_panel.csv"),
        os.path.join(os.getcwd(), "examples", "health_panel.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                return list(csv.DictReader(f))
    return None


def _project_health(
    panel: list[dict[str, Any]], iso: str, new_spending: float, mode: str, spec: dict[str, Any]
) -> dict[str, Any]:
    """Run the controlled-projection method for one country, using the ontology-sourced spec."""
    from .projection import controlled_projection

    outcome, dial, controls, log = spec["outcome"], spec["dial"], spec["controls"], spec["log_cols"]
    rows = [r for r in panel if r["iso3"] == iso]
    target = max(rows, key=lambda r: int(r["year"]))
    out: dict[str, Any] = {
        "country": target["country"],
        "iso3": iso,
        "year": target["year"],
        "current_spending": float(target[dial]),
        "current_under5_mortality": float(target[outcome]),
        "new_spending": new_spending,
        "panel_rows": len(panel),
        "ontology_validated": spec.get("ontology_validated", False),
    }
    if mode in ("both", "controlled"):
        out["controlled"] = controlled_projection(
            panel, outcome=outcome, dial=dial, controls=controls,
            target=target, new_dial=new_spending, log_cols=log,
            clamp_min=0.0,  # under-5 mortality cannot be negative
        )
    if mode in ("both", "naive"):
        out["naive"] = controlled_projection(
            panel, outcome=outcome, dial=dial, controls=[],
            target=target, new_dial=new_spending, log_cols=[dial],
            clamp_min=0.0,
        )
    return out


class AnswerRequest(BaseModel):
    """A natural-language question posted to /answer (the full chain)."""

    query_id: str
    question: str
    kb_id: str | None = None  # answer against a KB built via /build-kb; else the default world


class OntologyEditRequest(BaseModel):
    """A reviewer's edited ontology, submitted for CΩ validation."""

    ontology_yaml: str


class OntologyPublishRequest(BaseModel):
    """Human approval: freeze the reviewed ontology under a version decisions will cite."""

    version: str


class BuildFromDataRequest(BaseModel):
    """Sample rows posted to /build-kb-from-data: derive a draft ontology from existing data."""

    entity_type: str
    rows: list[dict[str, Any]]
    backing: str | None = None  # the table the rows came from, recorded on the entity


class BuildFromTablesRequest(BaseModel):
    """Several tables at once: ``{"tables": {"Order": [...], "OrderItem": [...]}}``.

    One table can only ever yield one entity. What connects them — relations, the field each is
    traversed by, cardinality, the subtype order — exists only across tables, so it needs all of
    them in one call.
    """

    tables: dict[str, list[dict[str, Any]]]
    backing: dict[str, str] | None = None  # entity type -> source table name


class ImpactRequest(BaseModel):
    """Tighten an action's guard and ask what loses eligibility, and what that reaches."""

    action: str
    new_threshold: float
    propagate_to: list[str] | None = None


class OntologyCompileRequest(BaseModel):
    """An externally-built ontology posted to /compile-ontology (→ W(q,t))."""

    ontology_id: str
    ontology_name: str = "ontology"
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]


def create_app(world: World | None = None) -> FastAPI:
    from .speechact import KB

    app = FastAPI(title="Loka Platform API", version="0.0.1")
    app.state.world = world or build_world_from_env()
    app.state.kb_worlds = {}  # kb_id -> World, populated by /build-kb
    app.state.kb = KB()  # KB.DATA / KB.METHODS; grows as queries are answered

    from .ontology_store import OntologyRecord, OntologyStore

    app.state.ontologies = OntologyStore()  # draft -> validated -> published lifecycle

    def _store() -> OntologyStore:
        """Typed access to the lifecycle store (app.state is untyped)."""
        store: OntologyStore = app.state.ontologies
        return store

    def _record_or_404(ontology_id: str) -> OntologyRecord:
        rec = _store().get(ontology_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"unknown ontology: {ontology_id}")
        return rec

    # Load the health ontology Ω and validate the projection method against it, so the method's
    # attributes are the ontology's attributes (the ontology is load-bearing, not decorative).
    from .scenario import load_health_ontology, method_spec

    # A method whose fields Ω does not declare must not run: that is what makes the ontology
    # load-bearing rather than descriptive, so the failure is raised, not absorbed. Only the
    # absence of an ontology altogether is tolerated — the method then reports itself as
    # unvalidated (ontology_validated: false) instead of pretending Ω authorised it.
    app.state.health_engine = load_health_ontology()
    app.state.health_method = method_spec(app.state.health_engine)

    # Governance context for the policy stage, sourced from Ω (version + action guard),
    # so the decision memo's enforced constraint is the ontology's, not a hardcoded string.
    _eng = app.state.health_engine
    app.state.health_governance = {
        "ontology_version": _eng.version if _eng else "unversioned",
        "guard": next((a.guard for a in _eng.action_types() if a.guard), "") if _eng else "",
    }

    @app.get("/health")
    def health() -> dict[str, str]:
        w: World = app.state.world
        return {
            "status": "ok",
            "ontology_version": w.engine.version,
            "backend": w.backend,
        }

    @app.post("/compile")
    def compile_endpoint(req: CompileRequest) -> dict[str, object]:
        w: World = app.state.world
        query = TypedQuery(
            query_id=req.query_id,
            task_type=req.task_type,
            targets=tuple(req.targets),
            signature=req.signature,
        )
        try:
            wqt = compile_wqt(
                w.engine, w.state, w.mission, query, scenario_id=req.query_id, causal=w.causal
            )
        except CompileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        encoded: dict[str, Any] = jsonable_encoder(wqt)
        return encoded

    @app.post("/build-kb")
    def build_kb_endpoint(req: BuildKBRequest) -> dict[str, Any]:
        """Workflow A: domain texts -> validated ontology + DATA/METHODS needs (a KBSpec).

        Follows texts->LLM->ontology when LOKA_LLM_BUILD is set and a model is
        configured (Claude or a self-hosted vLLM, via the model gateway); otherwise the
        deterministic rule-based builder. The response's ``builder`` field says which ran.

        What produced the draft is recorded on it: the model, the prompt verbatim, and a digest
        of the input. The prompt decides which parts of Ω can be extracted at all, so a draft
        without it cannot be reproduced or argued with — two runs over one document that differ
        because the prompt or the model differed would both be filed as "built by an LLM".
        """
        import hashlib
        import os
        import uuid

        from loka_ontology import OntologyLoadError, build

        from .world import world_from_kbspec

        if not req.texts:
            raise HTTPException(status_code=400, detail="no texts provided")

        spec = None
        builder_mode = "keyword"
        provenance: dict[str, Any] = {}

        llm_requested = os.getenv("LOKA_LLM_BUILD", "").lower() in ("1", "true", "yes")
        if llm_requested:
            # texts -> LLM -> ontology.
            try:
                from loka_ontology import LLMBuilder
                from loka_serving import llm_for, model_for

                llm_builder = LLMBuilder(
                    client=llm_for("ontology_build"),
                    model=model_for("ontology_build"),
                    system_prompt=req.system_prompt,
                )
                spec = build(req.texts, llm_builder)
                builder_mode = "llm"
                # Read back from the builder rather than restated here: a second copy of a
                # prompt is a copy that eventually stops matching the one that ran.
                provenance = {
                    "method": "text -> LLM -> ontology",
                    "model": llm_builder.model,
                    "prompt": llm_builder.system_prompt,
                    "prompt_source": "caller" if req.system_prompt else "default",
                }
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                # Falling back here used to look like graceful degradation. It is not: the
                # rule-based builder splits prose on capitalised words, so a real domain
                # document yields "What", "Two", "First", "UnderBrazilian" as entity types. That
                # went into the lifecycle as an ordinary draft, and a reader who did not check
                # `builder` would judge the extraction on twenty-four pieces of punctuation.
                #
                # A caller who set LOKA_LLM_BUILD asked for a model. Answering with something
                # else is answering a different question, so this reports the failure and its
                # cause instead. Unset the variable to ask for the rule-based builder on purpose.
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "the model-based build failed and no ontology was produced",
                        "cause": f"{type(exc).__name__}: {exc}",
                        "note": (
                            "LOKA_LLM_BUILD is set, so a model was requested. The rule-based "
                            "builder is not a substitute: it segments prose and would return "
                            "plausible-looking nonsense. Unset LOKA_LLM_BUILD to request it."
                        ),
                    },
                ) from exc

        if spec is None:
            try:
                spec = build(req.texts, None)
            except OntologyLoadError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            provenance = {
                # Named for what it is. It segments prose on capitalisation and produces
                # concept-shaped strings, which is enough to exercise the pipeline offline and
                # is not extraction; the name should not let a reader think otherwise.
                "method": "rule-based segmentation of the text (no model, not extraction)",
                "model": None,
                "prompt": None,
            }

        # The input is digested rather than stored: a domain document can be long and can be
        # confidential, and what a reader needs is to confirm they are looking at the same one.
        digest = hashlib.sha256("\n".join(req.texts).encode()).hexdigest()[:16]
        provenance["input_digest"] = digest
        provenance["input_texts"] = len(req.texts)
        provenance["input_chars"] = sum(len(t) for t in req.texts)

        kb_id = uuid.uuid4().hex[:12]
        app.state.kb_worlds[kb_id] = world_from_kbspec(spec)

        # The built ontology is a *proposal*: it enters the lifecycle as a draft and cannot
        # authorize an answer until a reviewer has edited it through CΩ and published it.
        rec = _store().create_draft(
            spec.ontology_yaml, source=f"builder:{builder_mode}", provenance=provenance
        )
        app.state.kb_worlds[kb_id].ontology_id = rec.ontology_id

        out: dict[str, Any] = jsonable_encoder(spec)
        out["kb_id"] = kb_id  # pass to /answer to query against this built KB
        out["builder"] = builder_mode  # 'llm' or 'keyword' (rule-based)
        out["ontology_id"] = rec.ontology_id
        out["state"] = rec.state  # 'draft' — not yet able to authorize answers
        out["source"] = rec.source
        # The prompt and model that produced this, returned with it rather than only stored:
        # a caller comparing two extractions needs the difference in front of them.
        out["provenance"] = dict(rec.provenance)
        out["review"] = rec.review  # what a human must decide before this can be published
        return out

    @app.post("/build-kb-from-data")
    def build_kb_from_data(req: BuildFromDataRequest) -> dict[str, Any]:
        """Derive a draft ontology from existing rows — the other way an ontology begins.

        Not every customer has a document describing their domain; most have tables. This reads
        sample rows and proposes an entity type with one typed property per column, recording the
        table it came from. What it infers from values is a guess — a date column read as text,
        a numeric code read as a number — so it enters the same lifecycle as a text-built
        ontology: a draft, with the review checklist naming what a machine reading data cannot
        settle, and no authority over an answer until a person publishes it.
        """
        from loka_ontology.infer import infer_ontology_from_rows, to_yaml

        if not req.rows:
            raise HTTPException(status_code=400, detail="no rows provided")
        try:
            onto = infer_ontology_from_rows(
                req.entity_type, req.rows, backing=req.backing
            )
        except Exception as exc:  # noqa: BLE001 - a malformed sample is a client error
            raise HTTPException(status_code=400, detail=f"could not infer: {exc}") from exc

        rec = _store().create_draft(to_yaml(onto), source=f"data:{req.backing or 'rows'}")
        out = rec.as_dict()
        out["rows_sampled"] = len(req.rows)
        return out

    @app.post("/build-kb-from-tables")
    def build_kb_from_tables(req: BuildFromTablesRequest) -> dict[str, Any]:
        """Derive a draft ontology from several tables — including what connects them.

        The single-table route proposes one entity and stops. Relations, the field each is
        walked by, cardinalities and the subtype order live between tables, and they are the part
        of Ω that makes it more than a schema. They are also decidable: a relation is proposed
        because one column's values are contained in another table's primary key, which is a fact
        that can be counted rather than a judgement that can be hallucinated.

        The response carries the evidence for every proposal — how many values resolved out of
        how many, how many were distinct, what the name similarity was — and the list of things
        this method cannot reach at all. Verbs, constraints, actions, guards and norms are
        decisions, not observations; a reviewer supplies them, and the draft says so rather than
        leaving their absence to be discovered later.
        """
        from loka_ontology.infer import to_yaml
        from loka_ontology.infer_tables import infer_ontology_from_tables

        if not req.tables:
            raise HTTPException(status_code=400, detail="no tables provided")
        empty = sorted(name for name, rows in req.tables.items() if not rows)
        if empty:
            raise HTTPException(status_code=400, detail=f"tables with no rows: {empty}")
        try:
            onto, report = infer_ontology_from_tables(req.tables, backing=req.backing)
        except Exception as exc:  # noqa: BLE001 - a malformed sample is a client error
            raise HTTPException(status_code=400, detail=f"could not infer: {exc}") from exc

        rec = _store().create_draft(
            to_yaml(onto), source=f"tables:{','.join(sorted(req.tables))}"
        )
        out = rec.as_dict()
        out["tables_read"] = {name: len(rows) for name, rows in req.tables.items()}
        out["inference"] = report.as_dict()
        return out

    @app.get("/ontology")
    def list_ontologies() -> dict[str, Any]:
        """Every ontology in the lifecycle, with its state and (once published) its version."""
        return {"ontologies": _store().list()}

    @app.get("/ontology/{ontology_id}")
    def get_ontology(ontology_id: str) -> dict[str, Any]:
        """The ontology YAML plus the review checklist — what a human must decide about it."""
        return _record_or_404(ontology_id).as_dict()

    @app.put("/ontology/{ontology_id}")
    def edit_ontology(ontology_id: str, req: OntologyEditRequest) -> dict[str, Any]:
        """Submit a reviewer's edit. Runs CΩ; on success the ontology becomes ``validated``.

        A CΩ failure returns 400 naming the rule that rejected it, so the reviewer sees what to
        fix rather than a generic error.
        """
        from .ontology_store import OntologyStateError

        _record_or_404(ontology_id)
        try:
            rec = _store().update(ontology_id, req.ontology_yaml)
        except OntologyStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - CΩ rejection is a client error, not a crash
            raise HTTPException(status_code=400, detail=f"CΩ rejected the ontology: {exc}") from exc
        return rec.as_dict()

    @app.post("/ontology/{ontology_id}/publish")
    def publish_ontology(ontology_id: str, req: OntologyPublishRequest) -> dict[str, Any]:
        """Human approval. Freezes the ontology under a version that decisions may cite."""
        from .ontology_store import OntologyStateError

        _record_or_404(ontology_id)
        try:
            rec = _store().publish(ontology_id, req.version)
        except OntologyStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # A world built at /build-kb holds the engine compiled from the *draft*. Leaving it there
        # would make review theatre: the gate would check the record's state while every query
        # still ran against the text the reviewer corrected. Rebind the published ontology, and
        # keep the state and evidence already ingested against it.
        from loka_ontology import OntologyEngine, load_ontology_str

        engine = OntologyEngine(load_ontology_str(rec.yaml))
        rebound = 0
        for world in app.state.kb_worlds.values():
            if getattr(world, "ontology_id", None) == ontology_id:
                world.engine = engine
                rebound += 1
        out = rec.as_dict()
        out["worlds_rebound"] = rebound
        return out

    def _panel_or_500() -> list[dict[str, Any]]:
        panel: list[dict[str, Any]] | None = app.state.__dict__.setdefault(
            "_health_panel", _load_health_panel()
        )
        if not panel:
            raise HTTPException(
                status_code=500, detail="health panel not found (set LOKA_HEALTH_PANEL)"
            )
        return panel

    @app.get("/scenario")
    def scenario_endpoint() -> dict[str, Any]:
        """Show the ontology Ω the scenario is bound to, and the ontology-validated method spec."""
        eng = app.state.health_engine
        attrs = (
            {n: str(p.base_type) for n, p in eng.properties_of("Country").items()} if eng else {}
        )
        return {
            "entity": "Country",
            "attributes": attrs,
            "method": app.state.health_method,  # includes ontology_validated: true/false
        }

    # ---- supply scenario: the relational half of Ω (R, ⪯, C) ----

    from .supply import load_supply_dataset, load_supply_ontology

    app.state.supply_engine = load_supply_ontology()
    app.state.supply_data = load_supply_dataset(app.state.supply_engine)

    def _relation_summaries(eng: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name, "from": r.from_type, "to": r.to_type,
                "via": r.via, "cardinality": r.cardinality,
            }
            for r in eng.relations()
        ]

    def _supply_or_503() -> Any:
        eng = app.state.supply_engine
        if eng is None:
            raise HTTPException(
                status_code=503, detail="supply ontology not found (set LOKA_SUPPLY_ONTOLOGY)"
            )
        return eng

    @app.get("/supply/scenario")
    def supply_scenario() -> dict[str, Any]:
        """The relational ontology: entities, the relations linking them, and the guarded actions.

        Where the health scenario has one entity and no relations, this one is where R, ⪯ and C
        carry weight — so it is the ontology to look at to see what Ω does beyond type-checking
        a single row.
        """
        from .supply import data_source

        eng = _supply_or_503()
        data = app.state.supply_data
        return {
            "ontology_version": eng.version,
            # Which rows these counts came from. A working answer over the wrong data looks
            # exactly like a working answer, so the source is reported rather than inferred
            # from the row counts by whoever happens to remember what the right ones are.
            "data_source": data_source(),
            "entities": {
                name: {
                    "properties": sorted(eng.properties_of(name)),
                    "subtype_of": (eng.supertypes(name) or [None])[0],
                    "rows": len(data.get(name, [])),
                }
                for name in eng.entity_types()
            },
            "relations": _relation_summaries(eng),
            # A = Au ⊎ Ac, and each action says which half it is in. Describing Ω is not the
            # same as proposing: an uncontrollable action belongs in the description of the
            # world, and is filtered out where actions are proposed.
            "actions": [
                {
                    "name": a.name,
                    "verb": a.verb,
                    "target": a.target,
                    "guard": a.guard,
                    "controllable": a.controllable,
                    "norms": [
                        {"name": n.name, "status": str(n.status), "when": n.when}
                        for n in eng.norms_for(a.name)
                    ],
                }
                for a in eng.action_types()
            ],
        }

    @app.get("/supply/route")
    def supply_route(from_type: str, to_type: str) -> dict[str, Any]:
        """The route Ω declares between two entity types — no join is written anywhere.

        Reports three distinct outcomes: a route, a route that reaches the target only by
        narrowing to a subtype (a runtime check the type system cannot make), or no route at all.
        """
        eng = _supply_or_503()
        for name in (from_type, to_type):
            if not eng.has_entity(name):
                raise HTTPException(
                    status_code=404,
                    detail=f"'{name}' is not an entity in ontology {eng.version}",
                )
        path = eng.path_between(from_type, to_type)
        narrowing = path is None and eng.needs_narrowing(from_type, to_type)
        if narrowing:
            path = eng.path_between(from_type, to_type, allow_narrowing=True)
        if path is None:
            return {
                "from": from_type, "to": to_type, "route": None,
                "reason": f"ontology {eng.version} declares no route between these types",
            }
        return {
            "from": from_type,
            "to": to_type,
            "hops": len(path),
            "route": [f"{r.name}{'>' if fwd else '<'}(via {r.via})" for r, fwd in path],
            "requires_narrowing": narrowing,
            "traversable": all(r.via for r, _ in path),
        }

    @app.post("/supply/impact")
    def supply_impact(req: ImpactRequest) -> dict[str, Any]:
        """Tighten a guard declared in Ω; report what loses eligibility and what that reaches.

        Every part of the answer comes from the ontology: the rule is an action's guard, the
        attribute it names must be declared on the action's target, subtypes are included via ⪯,
        and the consequence is followed along the declared relations by their declared link
        fields. Nothing here is a rule written in application code.
        """
        from .supply import impact_of_tightening

        eng = _supply_or_503()
        out: dict[str, Any] = impact_of_tightening(
            eng, app.state.supply_data,
            action_name=req.action, new_threshold=req.new_threshold,
            propagate_to=req.propagate_to,
        )
        if "error" in out:
            raise HTTPException(status_code=400, detail=out)
        return out

    @app.get("/kb")
    def kb_endpoint() -> dict[str, Any]:
        """Show KB.DATA (facts, growing as queries are answered) and KB.METHODS.

        ``data`` is the actual world only — observed facts, each with its provenance.
        ``all_facts`` additionally includes counterfactual worlds produced by ``orders`` acts,
        which are kept separate so a projection can never be read back as an observation.
        """
        kb = app.state.kb
        return {
            "data": kb.facts(),                          # actual world
            "all_facts": kb.facts(all_scenarios=True),   # + counterfactual worlds
            "methods": [
                {"name": m.name, "in_types": list(m.in_types), "out_type": m.out_type}
                for m in kb.methods.values()
            ],
        }

    @app.get("/methods")
    def methods_endpoint() -> dict[str, Any]:
        """What the system can compute — the other half of the knowledge base.

        A query is dispatched to KB.DATA or KB.METHODS, and one that names a method the KB does
        not hold is refused. ``GET /kb`` shows the facts; without this, the method half was only
        discoverable by asking for something and being told no.

        ``dispatch`` methods are those the typed-query path applies. ``registered`` are those the
        speech-act KB holds in this process; a method appears there once a request has caused it
        to be registered, so an empty list means none has been needed yet, not that none exists.
        """
        from .methods import method_catalog

        kb = app.state.kb
        return {
            "dispatch": method_catalog(),
            "registered": [
                {"name": m.name, "in_types": list(m.in_types), "out_type": m.out_type}
                for m in kb.methods.values()
            ],
        }

    @app.post("/project")
    def project_endpoint(req: ProjectRequest) -> dict[str, Any]:
        """Workflow B / KB.METHODS: project mortality when a country changes health spending."""
        panel = _panel_or_500()
        iso = req.country.upper()
        if not any(r["iso3"] == iso for r in panel):
            raise HTTPException(status_code=404, detail=f"unknown country: {req.country}")
        return _project_health(panel, iso, req.new_spending, req.mode, app.state.health_method)

    @app.post("/ask")
    def ask_endpoint(req: AskRequest) -> dict[str, Any]:
        """Full Workflow B: NL question -> LLM extracts {country, new_spending} -> project.

        If the question can't be grounded to the ontology (unknown country / no spending level),
        the system answers with an ``informs(li, sp, "don't know")`` act rather
        than guessing. This is the ontology doing its job: the system knows its own limits.
        """
        from .nl_project import as_spending, formalize_query, resolve_country
        from .scenario import ENTITY

        panel = _panel_or_500()
        spec = app.state.health_method
        engine = app.state.health_engine

        # Ω is the authority on which predicates exist. Both the LLM prompt and the validation
        # gate below are generated from it, so swapping the ontology swaps both.
        if engine is not None and engine.has_entity(ENTITY):
            omega_attrs = sorted(engine.properties_of(ENTITY))
            omega_version: str | None = engine.version
        else:
            omega_attrs = sorted({spec["outcome"], spec["dial"], *spec["controls"]})
            omega_version = None

        try:
            from loka_serving import llm_for, model_for

            proposal = formalize_query(
                req.question,
                llm_for("projection"),
                model_for("projection"),
                attributes=omega_attrs,
                entity=ENTITY,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"NL parsing needs an LLM (set OPENAI_*/ANTHROPIC_* + provider): {exc}",
            ) from exc

        from .speechact import (
            LISTENER,
            SPEAKER,
            Asks,
            Informs,
            Method,
            Orders,
            Provenance,
            dispatch,
        )

        outcome, dial = spec["outcome"], spec["dial"]
        kb = app.state.kb

        iso = resolve_country(panel, proposal.get("country"))
        spending = as_spending(proposal.get("new_spending"))
        attribute = proposal.get("attribute")

        def _dont_know(reason: str, code: str) -> dict[str, Any]:
            # informs(li, sp, "don't know") — the query could not be grounded in Ω / KB.
            return {
                "question": req.question,
                "answer": "don't know",
                "reason": reason,
                "reason_code": code,
                "ontology_version": omega_version,
                "formalized_query": proposal,
                "speech_act": {"act": "unformalizable", "query": None,
                               "response": Informs(LISTENER, SPEAKER, "don't know").render()},
            }

        if iso is None:
            return _dont_know(
                "the question named no country present in the panel", "unknown_entity"
            )

        # KB.METHODS: register the projection method m once (m[in,out] under5_mortality).
        if not kb.has_method("project_under5_mortality"):
            def _m(iso: str, new_spending: float) -> dict[str, Any]:
                res = _project_health(panel, iso, new_spending, "both", spec)
                return {"value": res.get("controlled", {}).get("projected_outcome"), "detail": res}

            kb.register_method(Method(
                name="project_under5_mortality",
                in_types=("Country", dial), out_type=outcome, fn=_m,
            ))

        # asks branch: a pure lookup of a current DATA attribute.
        if proposal.get("intent") == "ask" and spending is None:
            if not isinstance(attribute, str) or attribute not in omega_attrs:
                return _dont_know(
                    f"'{attribute}' is not a property of {ENTITY} in ontology "
                    f"{omega_version or '(none loaded)'}",
                    "not_in_ontology",
                )

            # Seed observed facts only now — a refused query must not mutate the KB.
            target = max((r for r in panel if r["iso3"] == iso), key=lambda r: int(r["year"]))
            vintage = str(target.get("year", ""))
            for attr in omega_attrs:
                if attr not in target:
                    continue
                try:
                    observed = float(target[attr])
                except (TypeError, ValueError):
                    continue
                kb.add_fact(
                    iso, attr, observed,
                    provenance=Provenance(
                        kind="observed", source="worldbank:WDI", vintage=vintage
                    ),
                )

            q_ask = Asks(SPEAKER, LISTENER, var_type=ENTITY, entity_id=iso, predicate=attribute)
            informs = dispatch(q_ask, kb)  # retrieve(d from KB.DATA), actual world only
            if not isinstance(informs.content, dict):
                return _dont_know(
                    f"'{attribute}' is declared in Ω but no observed value exists for {iso}",
                    "no_data",
                )
            content = informs.content
            return {
                "question": req.question,
                "answer": content.get("value"),
                "ontology_version": omega_version,
                "formalized_query": {"country": iso, "attribute": attribute},
                "retrieved": content,
                "speech_act": {
                    "act": "asks", "query": q_ask.render(), "response": informs.render(),
                },
            }

        # orders branch: change the dial, apply the projection method.
        if spending is None:
            return _dont_know(
                "no property to look up and no spending level to project", "unformalizable"
            )

        # q = orders(sp, li, m[in,out] P(x, m(x))) — apply the method, informs back, add P to KB.
        q = Orders(
            SPEAKER, LISTENER, method="project_under5_mortality",
            in_types=("Country", dial), out_type=outcome,
            entity_id=iso, predicate=outcome, args={"iso": iso, "new_spending": spending},
        )
        informs = dispatch(q, kb)  # applies m, writes the projected P back into KB.DATA

        result = informs.content["detail"] if isinstance(informs.content, dict) else {}
        result["question"] = req.question
        result["ontology_version"] = omega_version
        result["formalized_query"] = {"country": iso, "new_spending": spending}
        result["speech_act"] = {"act": "orders", "query": q.render(), "response": informs.render()}

        # the decision half: run the method result through Simulation -> Policy -> Decision memo.
        proj = result.get("controlled")
        if isinstance(proj, dict) and "projected_outcome" in proj:
            from .health_policy import evaluate_and_decide

            gov = app.state.health_governance
            rh = evaluate_and_decide(
                proj, iso=iso, ontology_version=gov["ontology_version"],
                guard=gov["guard"], method_name="project_under5_mortality",
            )
            result["scenarios"] = rh["scenarios"]
            result["decision"] = rh["decision"]
        return result

    @app.post("/kb/{kb_id}/ingest")
    def ingest_endpoint(kb_id: str, req: IngestRequest) -> dict[str, Any]:
        """Fill a built KB with data rows and causal claims, so /answer returns substance."""
        from .ingest import ingest_causal, ingest_data

        w = app.state.kb_worlds.get(kb_id)
        if w is None:
            raise HTTPException(status_code=404, detail=f"unknown kb_id: {kb_id}")
        stored, rejected = ingest_data(w, req.data)
        return {
            "kb_id": kb_id,
            "data_ingested": stored,
            # Values Ω's declared types do not admit, with the reason. Reported rather than
            # dropped: a caller that sent 40 rows and had 3 refused needs to know which.
            "data_rejected": rejected,
            "causal_ingested": ingest_causal(w, req.causal),
        }

    @app.post("/answer")
    def answer_endpoint(req: AnswerRequest) -> dict[str, Any]:
        """Full chain: NL question -> grounding -> W(q,t) -> simulation -> policy -> Response.

        Stages 2/3 are real; 4/5 are stubs (see the ``stages`` field of the response).
        """
        from .orchestrator import answer

        w: World = app.state.world
        if req.kb_id is not None:
            w = app.state.kb_worlds.get(req.kb_id)
            if w is None:
                raise HTTPException(status_code=404, detail=f"unknown kb_id: {req.kb_id}")

            # Only a published ontology may authorize an answer. A builder's proposal — however
            # well-formed — has not been reviewed, so it cannot stand behind a result.
            oid = getattr(w, "ontology_id", None)
            rec = _store().get(oid) if oid else None
            if rec is not None and rec.state != "published":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"ontology {oid} is '{rec.state}', not 'published': a generated ontology "
                        f"must pass CΩ and be approved before it can authorize an answer. "
                        f"{len(rec.review)} review item(s) outstanding — "
                        f"GET /ontology/{oid}, PUT the reviewed YAML, then POST .../publish."
                    ),
                )
        try:
            return answer(w, req.question, query_id=req.query_id)
        except CompileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/compile-ontology")
    def compile_ontology_endpoint(req: OntologyCompileRequest) -> dict[str, Any]:
        """Compile an externally-authored ontology into W(q,t) — used by Loka-OntoPrompt."""
        from .ontology_compile import compile_wqt_from_ontology

        if not req.entities:
            raise HTTPException(status_code=400, detail="ontology has no entities")
        try:
            return compile_wqt_from_ontology(
                req.entities,
                req.relations,
                ontology_id=req.ontology_id,
                ontology_name=req.ontology_name,
            )
        except CompileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
