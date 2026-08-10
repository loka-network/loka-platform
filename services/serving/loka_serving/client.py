"""Provider-agnostic LLM client — Claude, OpenAI, or a self-hosted vLLM endpoint.

Grounding and the ontology builder only need an object exposing ``messages.create(...)`` (the
Anthropic shape); they never care which backend serves it. ``make_llm_client()`` returns either
the Anthropic client or a thin adapter over an OpenAI-compatible endpoint (vLLM included), chosen
by env vars — so switching to a sovereign self-hosted model is configuration, not code:

    LOKA_LLM_PROVIDER = anthropic | openai | vllm
    LOKA_LLM_BASE_URL = http://host:8000/v1     (openai/vllm)
    LOKA_LLM_API_KEY  = ...                      (or "EMPTY" for local vLLM)
    LOKA_LLM_MODEL    = claude-opus-4-8 | Qwen3-... | <served model / lora name>
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any


def default_model() -> str:
    # LOKA_LLM_MODEL wins; else the standard OPENAI_MODEL (existing proxy setups); else default.
    return os.getenv("LOKA_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "claude-opus-4-8"


class _OpenAICompatMessages:
    """Adapts an OpenAI-compatible chat endpoint to the Anthropic ``messages.create`` shape."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **_: Any,
    ) -> Any:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        resp = self._client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=msgs
        )
        text = resp.choices[0].message.content or ""
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class OpenAICompatClient:
    """Exposes ``.messages.create(...)`` (Anthropic shape) over an OpenAI-compatible endpoint.

    ``client`` may be injected (for tests); otherwise an ``openai.OpenAI`` is built from base_url +
    api_key. A self-hosted vLLM server is OpenAI-compatible, and each LoRA adapter can be served
    under its own model name — so ``model=<lora-name>`` selects a persona for the behavior model.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import openai  # optional dependency, imported lazily

            client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.messages = _OpenAICompatMessages(client)


def make_llm_client() -> Any:
    """Return an LLM client.

    Selection order:
      1. ``LOKA_LLM_PROVIDER`` = openai | vllm  -> OpenAI-compatible;  = anthropic -> Anthropic.
      2. No provider set but an OpenAI-compatible endpoint is configured via the standard
         ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` (existing proxy setups, e.g. praka.ai) -> use it.
      3. Otherwise -> Anthropic direct.
    ``LOKA_LLM_*`` vars take precedence over the ``OPENAI_*`` ones when both are present.
    """
    provider = os.getenv("LOKA_LLM_PROVIDER", "").lower()
    base_url = os.getenv("LOKA_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("LOKA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider in ("openai", "vllm", "openai-compatible") or (not provider and base_url):
        return OpenAICompatClient(base_url=base_url, api_key=api_key)

    import anthropic  # optional dependency, imported lazily

    return anthropic.Anthropic()
