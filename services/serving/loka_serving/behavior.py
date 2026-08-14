"""Behavior engine port — where the company's human-behavior model plugs in (S4 Agent Society).

The simulation needs archetypes to behave realistically: given a social context, a persona, and
the interaction history, produce the persona's next action. That is exactly the behavior
foundation model's job (Qwen3-32B-Base + per-domain LoRA) — a DIFFERENT model and role than the
grounding LLM: not a helpful assistant but a de-assistantified persona simulator, with a LoRA
selected per persona/domain.

This module defines the port and two reference engines so the simulator can run before the real
model is ready:
  - ``StubBehaviorEngine``  — deterministic placeholder (no real behavior).
  - ``LLMBehaviorEngine``   — bridge over any LLM client (a stand-in; a helpful LLM is not a
                              faithful behavior simulator, but it lets the loop run end-to-end).
The real engine (a vLLM base_url with a LoRA per persona, via OpenAICompatClient) implements the
same ``act(...)`` signature — so wiring it in is filling an implementation, not changing the port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Persona:
    """A simulated actor. ``domain`` selects the LoRA adapter in the real behavior model."""

    name: str
    domain: str = "general"  # e.g. negotiation | support | central_bank — picks the LoRA
    traits: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class BehaviorEngine(Protocol):
    """Produce a persona's next action given the social context and history."""

    def act(self, *, social_context: str, persona: Persona, history: list[str]) -> str: ...


class StubBehaviorEngine:
    """Deterministic placeholder. Echoes a plausible action; NOT real behavior."""

    def act(self, *, social_context: str, persona: Persona, history: list[str]) -> str:
        last = history[-1] if history else "(opening move)"
        return f"[{persona.name}/{persona.domain}] acts in reply to: {last}"


class LLMBehaviorEngine:
    """Bridge: simulate behavior with a general LLM until the behavior model is served.

    Uses an injected client exposing ``messages.create(...)`` (Anthropic shape) — so the same
    ``OpenAICompatClient`` that points at a vLLM behavior-model endpoint works here too, with
    ``model`` set to the persona's LoRA name.
    """

    def __init__(self, *, client: Any, model: str = "claude-opus-4-8") -> None:
        self._client = client
        self._model = model

    def act(self, *, social_context: str, persona: Persona, history: list[str]) -> str:
        prompt = (
            f"Social context: {social_context}\n"
            f"You are: {persona.name} (domain: {persona.domain}); "
            f"traits: {', '.join(persona.traits) or 'none'}\n"
            f"History:\n" + ("\n".join(history) or "(none)") + "\n\n"
            "Reply with ONLY this persona's next action, in one line."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system="You act AS the given persona in a simulation. "
                   "Output only the action, no meta-commentary.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        return text.strip()
