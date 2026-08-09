"""The behavior engine port: a deterministic stub and an LLM bridge share one act() signature."""

from __future__ import annotations

from types import SimpleNamespace

from loka_serving.behavior import BehaviorEngine, LLMBehaviorEngine, Persona, StubBehaviorEngine


def test_stub_behavior_engine_conforms_to_port() -> None:
    eng = StubBehaviorEngine()
    assert isinstance(eng, BehaviorEngine)
    persona = Persona(name="Fed", domain="central_bank", traits=("cautious",))
    action = eng.act(social_context="rate meeting", persona=persona, history=["staff briefed"])
    assert "Fed" in action and "central_bank" in action


def test_llm_behavior_engine_uses_injected_client() -> None:
    def create(**_: object) -> object:
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hold rates steady")])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    eng = LLMBehaviorEngine(client=fake_client, model="Qwen3-32B-central_bank-lora")
    assert isinstance(eng, BehaviorEngine)
    action = eng.act(
        social_context="rate meeting",
        persona=Persona(name="Fed", domain="central_bank"),
        history=["inflation cooling"],
    )
    assert action == "hold rates steady"
