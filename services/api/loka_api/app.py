"""Loka Platform HTTP API (FastAPI).

Minimal service surface over the foundation:
  - GET  /health   → liveness + ontology version
  - POST /compile  → a typed query q* → the compiled Scenario World Model W(q, t) as JSON

The natural-language front-end (NL → q*) is S3; this endpoint takes an already-typed query.
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


class IngestRequest(BaseModel):
    """Data rows and causal claims to fill a built KB's DATA/METHODS needs."""

    data: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []


class AnswerRequest(BaseModel):
    """A natural-language question posted to /answer (the full slide-6 chain)."""

    query_id: str
    question: str
    kb_id: str | None = None  # answer against a KB built via /build-kb; else the default world


class OntologyCompileRequest(BaseModel):
    """An externally-built ontology posted to /compile-ontology (S1 + S2 → W(q,t))."""

    ontology_id: str
    ontology_name: str = "ontology"
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]


def create_app(world: World | None = None) -> FastAPI:
    app = FastAPI(title="Loka Platform API", version="0.0.1")
    app.state.world = world or build_world_from_env()
    app.state.kb_worlds = {}  # kb_id -> World, populated by /build-kb

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

        Follows the professor's texts->LLM->ontology when LOKA_LLM_BUILD is set and a model is
        configured (Claude or a self-hosted vLLM, via the model gateway); otherwise the
        deterministic rule-based builder. The response's ``builder`` field says which ran.
        """
        import os
        import uuid

        from loka_ontology import OntologyLoadError, build

        from .world import world_from_kbspec

        if not req.texts:
            raise HTTPException(status_code=400, detail="no texts provided")

        builder = None
        builder_mode = "keyword"
        if os.getenv("LOKA_LLM_BUILD", "").lower() in ("1", "true", "yes"):
            try:
                from loka_ontology import LLMBuilder
                from loka_serving import llm_for, model_for

                builder = LLMBuilder(
                    client=llm_for("ontology_build"), model=model_for("ontology_build")
                )
                builder_mode = "llm"
            except Exception:
                builder, builder_mode = None, "keyword"  # fall back if LLM unavailable

        try:
            spec = build(req.texts, builder)
        except OntologyLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        kb_id = uuid.uuid4().hex[:12]
        app.state.kb_worlds[kb_id] = world_from_kbspec(spec)
        out: dict[str, Any] = jsonable_encoder(spec)
        out["kb_id"] = kb_id  # pass to /answer to query against this built KB
        out["builder"] = builder_mode  # 'llm' (professor's way) or 'keyword' (rule-based)
        return out

    @app.post("/kb/{kb_id}/ingest")
    def ingest_endpoint(kb_id: str, req: IngestRequest) -> dict[str, Any]:
        """Fill a built KB with data rows and causal claims, so /answer returns substance."""
        from .ingest import ingest_causal, ingest_data

        w = app.state.kb_worlds.get(kb_id)
        if w is None:
            raise HTTPException(status_code=404, detail=f"unknown kb_id: {kb_id}")
        return {
            "kb_id": kb_id,
            "data_ingested": ingest_data(w, req.data),
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
