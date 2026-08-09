"""Manager Agent — the end-to-end query -> decision orchestrator.

This is the Sifakis slide-6 chain, wired for one question:

    question -> [3 grounding] q*  -> [2 compiler] W(q,t) -> [4 simulation] scenarios
             -> [5 policy] decision memorandum -> Response

Stages 2 (loka_compiler) and 3 (loka_grounding) run real code; stages 4 and 5 are honest stubs
(loka_api.simulation / loka_api.policy) until S4-S6 are built. The ``stages`` field of the
response says, per stage, whether it is real or a stub — no pretending.
"""

from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
from loka_compiler import compile_wqt
from loka_schemas import TypedQuery

from .methods import resolve
from .policy import decide
from .simulation import simulate
from .world import World


def answer(world: World, question: str, *, query_id: str) -> dict[str, Any]:
    """Run one natural-language question through the whole chain; return a Response dict."""
    # 3 · Semantic analysis / formalized query  (real when loka_grounding is installed)
    q_star, grounding_mode = _formalize(world, question, query_id=query_id)
    # 2 · Compile the per-question world model W(q, t)  (real)
    wqt = compile_wqt(
        world.engine,
        world.state,
        world.mission,
        q_star,
        scenario_id=query_id,
        causal=world.causal,
    )
    # 3b · Query dispatch — asks(DATA)->retrieve | orders(METHOD)->apply  (real)
    retrieval = resolve(q_star, wqt)
    # 4 · Simulate scenarios  (stub)
    scenarios = simulate(wqt)
    # 5 · Decide  (stub)
    memo = decide(wqt, scenarios)

    return {
        "query_id": query_id,
        "question": question,
        "formalized_query": jsonable_encoder(q_star),
        "retrieval": jsonable_encoder(retrieval),
        "world_model": jsonable_encoder(wqt),
        "scenarios": jsonable_encoder(scenarios),
        "decision": jsonable_encoder(memo),
        "stages": {
            "grounding": grounding_mode,
            "query_dispatch": "real",
            "compiler": "real",
            "causal": "real" if world.causal is not None else "empty",
            "simulation": "stub",
            "policy": "stub",
        },
    }


def _formalize(world: World, question: str, *, query_id: str) -> tuple[TypedQuery, str]:
    """3 · NL -> q*. Returns (q*, mode). Uses loka_grounding when available.

    Proposer selection: the LLM proposer runs only when ``LOKA_LLM_GROUNDING`` is set (and the
    'anthropic' extra + credentials resolve); otherwise the deterministic keyword proposer runs,
    so tests and offline/sovereign deployments stay reproducible. Either way the binder validates
    the proposal against the ontology, so a hallucinated entity is rejected, not trusted.
    """
    entity_types = _entity_types(world)
    try:
        from loka_grounding import ground
    except Exception:
        # Grounding package not importable in this env — keep the skeleton walking.
        targets = tuple(et for et in entity_types if et.lower() in question.lower())
        q = TypedQuery(query_id=query_id, task_type="descriptive", targets=targets, signature=None)
        return q, "fallback (grounding pkg absent)"

    proposer, mode = _make_proposer(entity_types)
    q_star: TypedQuery = ground(question, proposer, world.engine, query_id=query_id)
    return q_star, f"real ({mode})"


def _make_proposer(entity_types: list[str]) -> tuple[object, str]:
    """Pick the grounding proposer: LLM when opted in and available, else keyword reference."""
    import os

    from loka_grounding import KeywordProposer

    if os.getenv("LOKA_LLM_GROUNDING", "").lower() in ("1", "true", "yes"):
        try:
            from loka_grounding.llm_proposer import LLMProposer

            return LLMProposer(entity_types=entity_types), "llm"
        except Exception:
            pass  # fall through to the deterministic reference proposer
    return KeywordProposer(entity_types=entity_types), "keyword"


def _entity_types(world: World) -> list[str]:
    getter = getattr(world.engine, "entity_types", None)
    return list(getter()) if callable(getter) else []
