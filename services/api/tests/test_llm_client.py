"""The OpenAI-compatible adapter presents the Anthropic ``messages.create`` shape."""

from __future__ import annotations

from types import SimpleNamespace

from loka_api.llm_client import OpenAICompatClient


def test_openai_compat_adapter_wraps_response() -> None:
    # A fake OpenAI-style client (as vLLM would expose): chat.completions.create -> choices[0].message.content
    captured: dict[str, object] = {}

    def create(**kw: object) -> object:
        captured.update(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"task_type": "descriptive"}'))]
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
    assert captured["messages"][0] == {"role": "system", "content": "you are terse"}
    assert captured["model"] == "Qwen3-32B"
