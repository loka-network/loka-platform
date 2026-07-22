"""Stage ① — LLM-backed proposer (optional).

Uses Claude to turn a natural-language question into a candidate :class:`QueryProposal`.
This is the only place a language model appears in grounding, and it only *proposes* — the
binder still validates the result against the ontology, so a hallucinated entity is rejected,
not trusted. The ``anthropic`` package is an optional extra (``pip install
'loka-grounding[llm]'``); importing this module without it raises a clear error only when the
proposer is actually constructed.

For sovereign deployments the same interface can be pointed at a self-hosted model (e.g. via
a vLLM OpenAI-compatible endpoint) so no data leaves the customer's environment; the reference
implementation here calls the Anthropic API for development.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .models import TASK_TYPES, QueryProposal

# JSON schema the model must fill — structured output makes the response machine-parseable.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": sorted(TASK_TYPES)},
        "targets": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["task_type", "targets", "rationale"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You translate a user's economic-decision question into a structured query. "
    "Pick exactly one task_type from the allowed set and list the ontology entity types the "
    "question is about, using ONLY names from the provided list. Do not invent entity names; "
    "if the question mentions something not in the list, omit it. Return only the structured "
    "object."
)


class LLMProposer:
    """Claude-backed proposer. Constructed with the ontology's entity-type names."""

    def __init__(
        self,
        entity_types: Sequence[str],
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 512,
    ) -> None:
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "LLMProposer needs the 'anthropic' package: pip install 'loka-grounding[llm]'"
            ) from exc
        self._client = anthropic.Anthropic()  # resolves credentials from the environment
        self._entity_types = tuple(entity_types)
        self._model = model
        self._max_tokens = max_tokens

    def propose(self, question: str) -> QueryProposal:
        prompt = (
            f"Allowed entity types: {', '.join(self._entity_types)}\n"
            f"Allowed task types: {', '.join(sorted(TASK_TYPES))}\n\n"
            f"Question: {question}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text)
        return QueryProposal(
            task_type=str(data.get("task_type", "")),
            targets=tuple(str(t) for t in data.get("targets", [])),
            rationale=str(data.get("rationale", "")),
        )
