"""The model gateway routes by purpose, falls back to the behavior stub, and logs each call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loka_serving import Persona, StubBehaviorEngine, behavior_for
from loka_serving.gateway import audit_log, model_for


def test_model_for_default_and_per_purpose_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOKA_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOKA_LLM_MODEL_GROUNDING", raising=False)
    assert model_for("grounding") == "claude-opus-4-8"
    monkeypatch.setenv("LOKA_LLM_MODEL_GROUNDING", "Qwen3-32B")
    assert model_for("grounding") == "Qwen3-32B"


def test_behavior_for_falls_back_to_stub_when_no_model_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOKA_BEHAVIOR_BASE_URL", raising=False)
    monkeypatch.setattr("loka_serving.gateway.make_llm_client", _unavailable)
    eng, kind = behavior_for(Persona(name="Fed", domain="central_bank"))
    assert isinstance(eng, StubBehaviorEngine)
    assert kind == "stub"


def _unavailable() -> object:
    raise RuntimeError("no model configured")


def test_the_general_gateway_stands_in_but_is_not_called_a_behavior_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A general assistant is agreeable by construction, so a result from it must not be
    presented as a behavioural forecast. It is usable; it is labelled."""
    monkeypatch.delenv("LOKA_BEHAVIOR_BASE_URL", raising=False)
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: None))
    monkeypatch.setattr("loka_serving.gateway.make_llm_client", lambda: fake)
    _, kind = behavior_for(Persona(name="Fed", domain="central_bank"))
    assert kind == "general-llm"      # not "behavior-model"


def test_behavior_for_uses_injected_client_and_audits() -> None:
    fake = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(content=[SimpleNamespace(type="text", text="hold")])
        )
    )
    persona = Persona(name="Fed", domain="central_bank")
    eng, kind = behavior_for(persona, client=fake)
    assert kind == "behavior-model"
    assert eng.act(social_context="rate meeting", persona=persona, history=[]) == "hold"
    assert any(a["kind"] == "behavior" and a["purpose"] == "central_bank" for a in audit_log())
