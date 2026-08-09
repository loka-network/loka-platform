"""loka_serving — model gateway: the single governed boundary for LLM + behavior-model access.

Services ask by purpose (``llm_for``, ``behavior_for``) instead of constructing clients; provider,
model/LoRA routing, credentials, and an audit trail live here.
"""

from .behavior import BehaviorEngine, LLMBehaviorEngine, Persona, StubBehaviorEngine
from .client import OpenAICompatClient, default_model, make_llm_client
from .gateway import audit_log, behavior_for, llm_for, model_for

__all__ = [
    # client layer
    "make_llm_client",
    "OpenAICompatClient",
    "default_model",
    # behavior layer
    "BehaviorEngine",
    "Persona",
    "StubBehaviorEngine",
    "LLMBehaviorEngine",
    # gateway (route by purpose, audited)
    "llm_for",
    "model_for",
    "behavior_for",
    "audit_log",
]
