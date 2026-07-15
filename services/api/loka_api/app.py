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

from .world import World, build_default_world


class CompileRequest(BaseModel):
    """A typed query q* posted to /compile."""

    query_id: str
    task_type: str
    targets: list[str]
    signature: str | None = None


def create_app(world: World | None = None) -> FastAPI:
    app = FastAPI(title="Loka Platform API", version="0.0.1")
    app.state.world = world or build_default_world()

    @app.get("/health")
    def health() -> dict[str, str]:
        w: World = app.state.world
        return {"status": "ok", "ontology_version": w.engine.version}

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

    return app


app = create_app()
