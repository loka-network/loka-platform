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


def min_max_tokens() -> int:
    """Floor applied to ``max_tokens`` for OpenAI-compatible endpoints.

    Reasoning models (DeepSeek V4, o-series, …) spend the token budget on an internal chain of
    thought — returned in ``reasoning_content`` — *before* emitting any ``content``. A caller that
    asks for 200 tokens of JSON can therefore get an empty or truncated answer with
    ``finish_reason: length``, which surfaces as an unhelpful parse failure. The floor gives the
    reasoning pass headroom; a non-reasoning model simply stops early and is unaffected.
    """
    try:
        return int(os.getenv("LOKA_LLM_MIN_MAX_TOKENS", "1024"))
    except ValueError:
        return 1024


def max_tokens_ceiling() -> int:
    """The largest budget the escalating retry will ask for.

    How long a chain of thought runs is not knowable in advance and varies per call, so no
    constant is the right budget — the question is only where to stop escalating. This is that
    stopping point, not a target: a request that succeeds at 4k never asks for more.
    """
    try:
        return int(os.getenv("LOKA_LLM_MAX_TOKENS_CEILING", "32000"))
    except ValueError:
        return 32000


class EmptyCompletionError(RuntimeError):
    """The endpoint returned no content — typically the token budget went to reasoning."""


class TruncatedCompletionError(RuntimeError):
    """The reply was cut off by the token budget. Kept distinct from an empty one because the
    remedies differ: an empty reply means the reasoning consumed everything, a truncated one
    means the answer itself did not fit."""


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
        temperature: float | None = None,
        **_: Any,
    ) -> Any:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        budget = max(max_tokens, min_max_tokens())

        # A reasoning model's chain of thought varies in length from call to call, and on a long
        # document it can consume the whole budget before a single character of the answer is
        # emitted — the reply comes back empty with finish_reason=length. Doubling once was not
        # enough: a 4.8k-character domain text produced 33k characters of reasoning and still
        # nothing at 8000. So the budget escalates until the model answers or the ceiling is
        # reached, and only a length-truncated reply is retried; any other empty reply means
        # something else is wrong and more tokens will not fix it.
        ceiling = max_tokens_ceiling()
        attempts = 0
        while True:
            attempts += 1
            extra = {} if temperature is None else {"temperature": temperature}
            resp = self._client.chat.completions.create(
                model=model, max_tokens=budget, messages=msgs, **extra
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish = getattr(choice, "finish_reason", "?")

            # ``length`` means the budget ran out, and it means that whether or not anything was
            # emitted first. A half-written reply is the more dangerous half of that: it looks
            # like an answer, so it used to be returned, and the caller met it as a parse error
            # pointing at whatever character the model happened to stop on — a message about
            # JSON syntax for a problem that has nothing to do with syntax.
            if finish == "length" and budget < ceiling:
                budget = min(budget * 2, ceiling)
                continue
            if text.strip() and finish != "length":
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

            reasoning = getattr(choice.message, "reasoning_content", None)
            where = f"budget={budget}, ceiling={ceiling}"
            if text.strip():
                # Truncated at the ceiling. Say so, and show where it stopped: a reader can tell
                # a cut-off structure from a model that was answering the wrong question.
                raise TruncatedCompletionError(
                    f"model {model} was still truncated at the ceiling after {attempts} "
                    f"attempt(s) ({where}, {len(text)} chars returned"
                    + (f", {len(reasoning)} chars of reasoning" if reasoning else "")
                    + f"); the reply ends: ...{text[-160:]!r}. Raise "
                    "LOKA_LLM_MAX_TOKENS_CEILING, shorten the input, or ask for less output"
                )
            # Nothing at all came back: the budget went entirely to the chain of thought.
            raise EmptyCompletionError(
                f"model {model} returned no content after {attempts} attempt(s) "
                f"(finish_reason={finish}, {where}"
                + (f", {len(reasoning)} chars of reasoning" if reasoning else "")
                + "); raise LOKA_LLM_MAX_TOKENS_CEILING, or use a model that does not spend "
                "the whole budget reasoning"
            )


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
