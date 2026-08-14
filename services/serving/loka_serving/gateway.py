"""Model gateway — the single governed boundary for all model access.

Every service asks the gateway for a model *by purpose* rather than constructing clients itself,
so provider / model / LoRA routing, credentials, and an audit trail live in one place — the
boundary between what a human controls and what the machine does.
Grounding and the ontology builder still accept an injected client; the gateway is just who
constructs, routes, and logs it.

Config:
  general LLM      — LOKA_LLM_PROVIDER / BASE_URL / API_KEY / MODEL (see client.py);
                     per-purpose override: LOKA_LLM_MODEL_<PURPOSE>
  behavior model   — LOKA_BEHAVIOR_BASE_URL / API_KEY, LOKA_BEHAVIOR_MODEL_TMPL (default "{domain}")
                     so a persona's domain selects its served LoRA name.
"""

from __future__ import annotations

import os
from typing import Any

from .behavior import BehaviorEngine, LLMBehaviorEngine, Persona, StubBehaviorEngine
from .client import OpenAICompatClient, default_model, make_llm_client

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


def behavior_for(persona: Persona, *, client: Any | None = None) -> BehaviorEngine:
    """Return a behavior engine for a persona.

    Real deployment: a vLLM behavior-model endpoint (LOKA_BEHAVIOR_BASE_URL) with the persona's
    domain selecting its LoRA. If no client/endpoint is available, fall back to the deterministic
    stub so the simulator still runs.
    """
    base = os.getenv("LOKA_BEHAVIOR_BASE_URL")
    if client is None and base:
        client = OpenAICompatClient(base_url=base, api_key=os.getenv("LOKA_BEHAVIOR_API_KEY"))
    if client is not None:
        model = os.getenv("LOKA_BEHAVIOR_MODEL_TMPL", "{domain}").format(domain=persona.domain)
        _log("behavior", persona.domain, model)
        return LLMBehaviorEngine(client=client, model=model)
    _log("behavior", persona.domain, "stub")
    return StubBehaviorEngine()
