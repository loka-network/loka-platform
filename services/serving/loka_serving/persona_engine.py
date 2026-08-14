"""Persona engine port — where a domain human-behaviour model plugs in.

NOT the agent's behaviour. The word is overloaded and the two meanings point in opposite
directions, so this module does not use it:

  * *Behaviour* in the agent-architecture sense is the transition system ``B(s0) = (s0, S, →)``
    with ``→ ⊆ S × A × S`` — this system's OWN reachable states, over which invariants and
    optimisation goals are stated. Loka does not build that object; see the simulation stage.
  * A *persona engine* produces the next action of an actor being SIMULATED — someone else, in a
    virtual environment, not this system.

Own behaviour versus simulated others' behaviour: reading a persona engine as the former would
mean reading a stand-in for a counterparty as a statement about what Loka itself will do.

The simulation needs archetypes to act plausibly: given a social context, a persona, and the
interaction history, produce that persona's next action. That is the behaviour foundation
model's job (Qwen3-32B-Base + per-domain LoRA) — a different model and role from the grounding
LLM: not a helpful assistant but a de-assistantified persona simulator, LoRA-selected per domain.

Two reference engines let the loop run before the real model is served:
  - ``StubPersonaEngine``  — deterministic placeholder (no behaviour at all).
  - ``LLMPersonaEngine``   — bridge over any LLM client (a stand-in; a helpful assistant is not a
                             faithful persona simulator, but it closes the loop end-to-end).
The real engine (a vLLM base_url with a LoRA per persona, via OpenAICompatClient) implements the
same ``act(...)`` signature — so wiring it in fills an implementation, it does not change the port.

The ``LOKA_BEHAVIOR_*`` environment variables keep their names: they configure the *behaviour
foundation model*, which is what that product is called, and renaming them would break deployed
configuration for a distinction that lives in this codebase rather than in the model.
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
class PersonaEngine(Protocol):
    """Produce a persona's next action given the social context and history."""

    def act(self, *, social_context: str, persona: Persona, history: list[str]) -> str: ...


class StubPersonaEngine:
    """Deterministic placeholder. Echoes a plausible action; NOT real behavior."""

    def act(self, *, social_context: str, persona: Persona, history: list[str]) -> str:
        last = history[-1] if history else "(opening move)"
        return f"[{persona.name}/{persona.domain}] acts in reply to: {last}"


class LLMPersonaEngine:
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
