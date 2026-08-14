"""Model gateway — the single governed boundary for all model access.

Every service asks the gateway for a model *by purpose* rather than constructing clients itself,
so provider / model / LoRA routing, credentials, and an audit trail live in one place — the
boundary between what a human controls and what the machine does.
Grounding and the ontology builder still accept an injected client; the gateway is just who
constructs, routes, and logs it.

Config:
  general LLM      — LOKA_LLM_PROVIDER / BASE_URL / API_KEY / MODEL (see client.py);
                     per-purpose override: LOKA_LLM_MODEL_<PURPOSE>
  behaviour model  — LOKA_BEHAVIOR_BASE_URL / API_KEY, LOKA_BEHAVIOR_MODEL_TMPL (default "{domain}")
                     so a persona's domain selects its served LoRA name. The variables name the
                     external behaviour foundation model; inside this codebase the port it backs
                     is the *persona* engine, which simulates other actors (see persona_engine).
"""

from __future__ import annotations

import os
from typing import Any

from .client import OpenAICompatClient, default_model, make_llm_client
from .persona_engine import LLMPersonaEngine, Persona, PersonaEngine, StubPersonaEngine

_AUDIT: list[dict[str, str]] = []


def _log(kind: str, purpose: str, model: str) -> None:
    _AUDIT.append({"kind": kind, "purpose": purpose, "model": model})


def audit_log() -> list[dict[str, str]]:
    """Every model resolution the gateway handed out (for replay / governance)."""
    return list(_AUDIT)


def model_for(purpose: str) -> str:
    """The model name for a purpose; per-purpose override else the default."""
    return os.getenv(f"LOKA_LLM_MODEL_{purpose.upper()}", default_model())


def llm_for(purpose: str) -> Any:
    """Return a governed LLM client for a purpose (e.g. 'grounding', 'ontology_build')."""
    client = make_llm_client()
    _log("llm", purpose, model_for(purpose))
    return client


def persona_engine_for(
    persona: Persona, *, client: Any | None = None
) -> tuple[PersonaEngine, str]:
    """Return a persona engine, and what kind of engine it is.

    Three cases, and the caller is told which it got, because they are not interchangeable:

      ``behavior-model``   a served behaviour model (LOKA_BEHAVIOR_BASE_URL), the persona's domain
                           selecting its adapter. This is the one trained to act *as* a persona.
      ``general-llm``      the ordinary model gateway, standing in. A general assistant is
                           agreeable by construction, so it under-produces the refusals, delays
                           and adversarial moves a simulation exists to find. Usable, but a
                           result from it must not be read as a calibrated behavioural forecast.
      ``stub``             deterministic placeholder; no behaviour at all.

    Returning the kind alongside the engine is what lets an output say which one produced it,
    rather than presenting all three as the same thing.
    """
    base = os.getenv("LOKA_BEHAVIOR_BASE_URL")
    if client is None and base:
        client = OpenAICompatClient(base_url=base, api_key=os.getenv("LOKA_BEHAVIOR_API_KEY"))
        model = os.getenv("LOKA_BEHAVIOR_MODEL_TMPL", "{domain}").format(domain=persona.domain)
        _log("behavior", persona.domain, model)
        return LLMPersonaEngine(client=client, model=model), "behavior-model"
    if client is not None:
        model = os.getenv("LOKA_BEHAVIOR_MODEL_TMPL", "{domain}").format(domain=persona.domain)
        _log("behavior", persona.domain, model)
        return LLMPersonaEngine(client=client, model=model), "behavior-model"

    try:  # no dedicated endpoint — stand in with the general gateway, and say so
        general = make_llm_client()
        _log("behavior", persona.domain, f"{default_model()} (stand-in)")
        return LLMPersonaEngine(client=general, model=default_model()), "general-llm"
    except Exception:  # noqa: BLE001 - no model reachable at all
        _log("behavior", persona.domain, "stub")
        return StubPersonaEngine(), "stub"
