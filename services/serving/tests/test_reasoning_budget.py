"""Reasoning models spend the budget on reasoning_content before emitting content."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from loka_serving.client import EmptyCompletionError, OpenAICompatClient, min_max_tokens


class _Rec:
    def __init__(self, content: str, reasoning: str = "") -> None:
        self.seen: dict[str, object] = {}
        self._c, self._r = content, reasoning
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: object) -> object:
        self.seen = kw
        msg = SimpleNamespace(content=self._c, reasoning_content=self._r)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])


def test_max_tokens_is_floored_for_reasoning_models() -> None:
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=200, messages=[])
    assert rec.seen["max_tokens"] == min_max_tokens() >= 1024  # 200 would be eaten by reasoning


def test_caller_may_ask_for_more_than_the_floor() -> None:
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=8000, messages=[])
    assert rec.seen["max_tokens"] == 8000


def test_empty_content_raises_an_actionable_error() -> None:
    rec = _Rec("", reasoning="thinking..." * 50)
    with pytest.raises(EmptyCompletionError, match="no content"):
        OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=200, messages=[])
