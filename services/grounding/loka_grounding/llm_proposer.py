"""Stage ① — LLM-backed proposer (optional).

Uses a chat model to turn a natural-language question into a candidate ``QueryProposal``.
This is the only place a language model appears in grounding, and it only *proposes* — the
binder still validates the result against the ontology, so a hallucinated entity is rejected,
not trusted.

The model client is injectable: by default it constructs ``anthropic.Anthropic()`` (the
``anthropic`` package is an optional ``[llm]`` extra, imported lazily), but any object exposing
``messages.create(...)`` works — a fake in tests, or a self-hosted endpoint (e.g. vLLM) for a
sovereign deployment where no data may leave the customer's environment. The response is parsed
as plain JSON rather than a provider-specific structured-output format, so the same proposer
works across model backends.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .models import TASK_TYPES, QueryProposal

_SYSTEM = (
    "You translate a user's economic-decision question into a structured query. "
    "Reply with ONLY a JSON object of the form "
    '{"task_type": <one of ' + str(sorted(TASK_TYPES)) + ">, "
    '"targets": [<ontology entity type names>], "rationale": <short string>}. '
    "Use ONLY entity-type names from the provided list; if the question mentions something "
    "not in the list, omit it. Do not wrap the JSON in prose or code fences."
)


def _parse_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in model reply: {text!r}")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model reply is not a JSON object")
    return obj


def _reply_text(resp: Any) -> str:
    """Concatenate the text blocks of a messages response."""
    parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"]
    return "".join(parts)


class LLMProposer:
    """Model-backed proposer. Give it the ontology's entity-type names."""

    def __init__(
        self,
        entity_types: Sequence[str],
        *,
        client: Any | None = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 512,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "LLMProposer needs 'anthropic' (pip install 'loka-grounding[llm]') "
                    "or an injected client exposing messages.create(...)"
                ) from exc
            client = anthropic.Anthropic()  # resolves credentials from the environment
        self._client = client
        self._entity_types = tuple(entity_types)
        self._model = model
        self._max_tokens = max_tokens

    def propose(self, question: str) -> QueryProposal:
        prompt = (
            f"Allowed entity types: {', '.join(self._entity_types)}\n\n"
            f"Question: {question}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        data = _parse_json_object(_reply_text(resp))
        targets = data.get("targets", [])
        return QueryProposal(
            task_type=str(data.get("task_type", "")),
            targets=tuple(str(t) for t in targets) if isinstance(targets, list) else (),
            rationale=str(data.get("rationale", "")),
        )
