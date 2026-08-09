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


def test_behavior_for_falls_back_to_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOKA_BEHAVIOR_BASE_URL", raising=False)
    eng = behavior_for(Persona(name="Fed", domain="central_bank"))
    assert isinstance(eng, StubBehaviorEngine)


def test_behavior_for_uses_injected_client_and_audits() -> None:
    fake = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(content=[SimpleNamespace(type="text", text="hold")])
        )
    )
    persona = Persona(name="Fed", domain="central_bank")
    eng = behavior_for(persona, client=fake)
    assert eng.act(social_context="rate meeting", persona=persona, history=[]) == "hold"
    assert any(a["kind"] == "behavior" and a["purpose"] == "central_bank" for a in audit_log())
