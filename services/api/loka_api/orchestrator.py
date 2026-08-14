"""Manager Agent — the end-to-end query -> decision orchestrator.

The chain, wired for one question:

    question -> [3 grounding] q*  -> [2 compiler] W(q,t) -> [4 simulation] scenarios
             -> [5 policy] decision memorandum -> Response

Stages 2 (loka_compiler) and 3 (loka_grounding) run real code; stages 4 and 5 are honest stubs
(loka_api.simulation / loka_api.policy) until the simulator and policy model are built.
The ``stages`` field of the
response says, per stage, whether it is real or a stub — no pretending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.encoders import jsonable_encoder
from loka_compiler import compile_wqt
from loka_schemas import TypedQuery

from .actions import propose_actions
from .methods import resolve
from .policy import decide
from .simulation import actor_reactions, simulate
from .world import World

if TYPE_CHECKING:  # the grounding package is an optional dependency at runtime
    from loka_grounding.proposer import QueryProposer


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
    # 4 · Simulate scenarios  (basic: causal-driven when Γ(q) has claims, else canned)
    scenarios = simulate(wqt)
    # 5 · Decide  (basic: welfare + hard-constraint gate)
    memo = decide(wqt, scenarios)
    # 5b · Action layer — governed action proposals (guard + G3 gate, needs confirmation)
    actions = propose_actions(world, wqt)
    has_causal_claims = bool(wqt.causal_slice and wqt.causal_slice.claims)

    # An answer resting on a claim whose sources disagree beyond sampling error is not the same
    # as one whose sources agree, and the pooled estimate alone cannot show the difference.
    from .ingest import contradictions_for

    cited = [c.claim_id for c in wqt.causal_slice.claims] if wqt.causal_slice else []
    disagreements = contradictions_for(world, cited)

    # 4b · How the actors Ω permits would respond to the downside. Reported with the kind of
    # engine that produced it, since a stand-in and a trained behavior model are not the same
    # evidence.
    adverse = next((s for s in scenarios if s.kind == "adverse"), None)
    reactions = actor_reactions(
        world.engine, str(adverse.outcome) if adverse is not None else "baseline"
    )

    return {
        "query_id": query_id,
        "question": question,
        "formalized_query": jsonable_encoder(q_star),
        "retrieval": jsonable_encoder(retrieval),
        "world_model": jsonable_encoder(wqt),
        "scenarios": jsonable_encoder(scenarios),
        "decision": jsonable_encoder(memo),
        "actions": jsonable_encoder(actions),
        "evidence_conflicts": disagreements,
        "actor_reactions": reactions,
        "stages": {
            "grounding": grounding_mode,
            "query_dispatch": "real",
            "compiler": "real",
            "causal": "real" if world.causal is not None else "empty",
            "evidence": ("conflicted" if disagreements else "consistent") if cited else "none",
            "simulation": "basic" if has_causal_claims else "stub",
            "behavior": reactions["engine"],
            "policy": "basic",
            "action": "basic" if actions else "none",
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


def _make_proposer(entity_types: list[str]) -> tuple[QueryProposer, str]:
    """Pick the grounding proposer: LLM when opted in and available, else keyword reference."""
    import os

    from loka_grounding import KeywordProposer

    if os.getenv("LOKA_LLM_GROUNDING", "").lower() in ("1", "true", "yes"):
        try:
            from loka_grounding.llm_proposer import LLMProposer
            from loka_serving import llm_for, model_for

            proposer = LLMProposer(
                entity_types=entity_types,
                client=llm_for("grounding"),
                model=model_for("grounding"),
            )
            return proposer, "llm"
        except Exception as exc:  # noqa: BLE001 - degrade, but say why
            # The operator asked for the model proposer, so falling back to the deterministic
            # one is a downgrade they need to see: a silent switch would make a run that used a
            # different proposer indistinguishable from one that used the requested one.
            return (
                KeywordProposer(entity_types=entity_types),
                f"keyword (LLM proposer requested but unavailable: {type(exc).__name__}: {exc})",
            )
    return KeywordProposer(entity_types=entity_types), "keyword"


def _entity_types(world: World) -> list[str]:
    getter = getattr(world.engine, "entity_types", None)
    return list(getter()) if callable(getter) else []
