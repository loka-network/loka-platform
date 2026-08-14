"""The OpenAI-compatible adapter presents the Anthropic ``messages.create`` shape."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loka_serving.client import OpenAICompatClient, default_model, make_llm_client


def test_openai_compat_adapter_wraps_response() -> None:
    # A fake OpenAI-style client (as vLLM exposes): chat.completions.create -> choices[0]
    captured: dict[str, object] = {}

    def create(**kw: object) -> object:
        captured.update(kw)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"task_type": "descriptive"}'))
            ]
        )

    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client = OpenAICompatClient(client=fake_openai)

    resp = client.messages.create(
        model="Qwen3-32B",
        max_tokens=64,
        system="you are terse",
        messages=[{"role": "user", "content": "hi"}],
    )
    # Anthropic-shaped response so grounding/builder need no changes.
    assert resp.content[0].type == "text"
    assert resp.content[0].text == '{"task_type": "descriptive"}'
    # system was folded into messages for the OpenAI schema.
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "you are terse"}
    assert captured["model"] == "Qwen3-32B"


def test_make_llm_client_picks_up_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Existing proxy setup (OPENAI_* vars, e.g. praka.ai) is used without any LOKA_LLM_* vars.
    monkeypatch.delenv("LOKA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LOKA_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://praka.ai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(make_llm_client(), OpenAICompatClient)


def test_default_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOKA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert default_model() == "claude-opus-4-8"
    monkeypatch.setenv("OPENAI_MODEL", "claude-opus-4-6")
    assert default_model() == "claude-opus-4-6"
    monkeypatch.setenv("LOKA_LLM_MODEL", "Qwen3-32B")
    assert default_model() == "Qwen3-32B"
