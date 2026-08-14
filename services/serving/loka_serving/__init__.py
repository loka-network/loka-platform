"""loka_serving — model gateway: the single governed boundary for LLM + behaviour-model access.

Services ask by purpose (``llm_for``, ``persona_engine_for``) instead of constructing clients;
provider, model/LoRA routing, credentials, and an audit trail live here.

``persona_engine`` simulates *other* actors. It is deliberately not called "behaviour": that word
is reserved for the agent's own transition system, which this system does not build.
"""

from .client import OpenAICompatClient, default_model, make_llm_client
from .gateway import audit_log, llm_for, model_for, persona_engine_for
from .persona_engine import LLMPersonaEngine, Persona, PersonaEngine, StubPersonaEngine

__all__ = [
    # client layer
    "make_llm_client",
    "OpenAICompatClient",
    "default_model",
    # persona layer — engines that simulate OTHER actors, not this agent's own behaviour
    "PersonaEngine",
    "Persona",
    "StubPersonaEngine",
    "LLMPersonaEngine",
    # gateway (route by purpose, audited)
    "llm_for",
    "model_for",
    "persona_engine_for",
    "audit_log",
]
