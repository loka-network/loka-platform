"""Reasoning models spend the budget on reasoning_content before emitting content."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from loka_serving.client import (
    EmptyCompletionError,
    OpenAICompatClient,
    TruncatedCompletionError,
    min_max_tokens,
)


class _Rec:
    """``finish_reason`` defaults to ``stop``: a reply that is complete did not run out of
    budget. Pairing full content with ``length`` describes a state no endpoint produces, and a
    fixture that does so tests behaviour against a reality that does not exist."""

    def __init__(self, content: str, reasoning: str = "", finish: str = "stop") -> None:
        self.seen: dict[str, object] = {}
        self._c, self._r, self._f = content, reasoning, finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: object) -> object:
        self.seen = kw
        msg = SimpleNamespace(content=self._c, reasoning_content=self._r)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self._f)])


def test_max_tokens_is_floored_for_reasoning_models() -> None:
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=200, messages=[])
    assert rec.seen["max_tokens"] == min_max_tokens() >= 1024  # 200 would be eaten by reasoning


def test_caller_may_ask_for_more_than_the_floor() -> None:
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=8000, messages=[])
    assert rec.seen["max_tokens"] == 8000


def test_empty_content_raises_an_actionable_error() -> None:
    rec = _Rec("", reasoning="thinking..." * 50, finish="length")
    with pytest.raises(EmptyCompletionError, match="no content"):
        OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=200, messages=[])


def test_an_exhausted_budget_is_retried_with_double() -> None:
    """A reasoning model's chain of thought varies per call, so a budget that sufficed once can
    be consumed entirely by the next request. Doubling adapts; a larger constant would only move
    the threshold at which it recurs."""
    class _Flaky:
        def __init__(self) -> None:
            self.budgets: list[object] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw: object) -> object:
            self.budgets.append(kw["max_tokens"])
            first = len(self.budgets) == 1
            msg = SimpleNamespace(
                content="" if first else "ok", reasoning_content="thinking" * 200
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=msg, finish_reason="length" if first else "stop")
                ]
            )

    rec = _Flaky()
    resp = OpenAICompatClient(client=rec).messages.create(
        model="m", max_tokens=4000, messages=[]
    )
    assert resp.content[0].text == "ok"
    assert rec.budgets == [4000, 8000]      # doubled, not a bigger constant


def test_a_half_written_reply_is_retried_rather_than_returned() -> None:
    """The bug this was written for. A truncated reply is not empty, so it used to be returned
    as though it were an answer — and the caller met it as a JSON syntax error pointing at
    whatever character the model stopped on. The syntax was never the problem."""
    class _CutOff:
        def __init__(self) -> None:
            self.budgets: list[object] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw: object) -> object:
            self.budgets.append(kw["max_tokens"])
            first = len(self.budgets) == 1
            msg = SimpleNamespace(content='{"entities": [{"na' if first else '{"entities": []}')
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=msg, finish_reason="length" if first else "stop")
                ]
            )

    rec = _CutOff()
    resp = OpenAICompatClient(client=rec).messages.create(
        model="m", max_tokens=4000, messages=[]
    )
    assert resp.content[0].text == '{"entities": []}'   # the complete one, not the fragment
    assert rec.budgets == [4000, 8000]


def test_a_reply_still_truncated_at_the_ceiling_says_so_and_shows_where(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When more budget is not the answer, the operator needs the text — a character offset into
    something they cannot see does not tell them whether to shorten the input or the output."""
    monkeypatch.setenv("LOKA_LLM_MAX_TOKENS_CEILING", "4000")
    rec = _Rec('{"entities": [{"name": "Selle', finish="length")
    with pytest.raises(TruncatedCompletionError, match="still truncated at the ceiling"):
        OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=4000, messages=[])


def test_the_budget_escalates_until_the_ceiling_then_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doubling once was not enough. A 4.8k-character domain document made a reasoning model
    emit 33k characters of thought and no answer at a budget of 8000, so the request failed and
    the caller was handed a fallback they had not asked for. The budget now climbs until the
    model answers or the ceiling is reached — the ceiling being where to stop escalating, not a
    target: a call that succeeds at the first budget never asks for more.
    """
    monkeypatch.setenv("LOKA_LLM_MIN_MAX_TOKENS", "1000")
    monkeypatch.setenv("LOKA_LLM_MAX_TOKENS_CEILING", "8000")

    class _Always:
        def __init__(self) -> None:
            self.budgets: list[int] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw: object) -> object:
            self.budgets.append(int(kw["max_tokens"]))  # type: ignore[call-overload]
            msg = SimpleNamespace(content="", reasoning_content="x" * 50)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    rec = _Always()
    with pytest.raises(EmptyCompletionError, match="ceiling=8000"):
        OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=1000, messages=[])
    assert rec.budgets == [1000, 2000, 4000, 8000]  # doubling, and it stops at the ceiling


def test_a_budget_that_works_is_not_escalated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escalation is a response to truncation, not a habit. Asking every model for the ceiling
    would make every call cost what the worst call costs."""
    monkeypatch.setenv("LOKA_LLM_MIN_MAX_TOKENS", "1000")

    class _Answers:
        def __init__(self) -> None:
            self.budgets: list[int] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw: object) -> object:
            self.budgets.append(int(kw["max_tokens"]))  # type: ignore[call-overload]
            msg = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    rec = _Answers()
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=1000, messages=[])
    assert rec.budgets == [1000]


def test_an_empty_answer_that_was_not_truncated_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finish_reason=stop with no content means the model chose to say nothing. More tokens
    will not change that, and retrying would turn one wasted call into several."""
    monkeypatch.setenv("LOKA_LLM_MAX_TOKENS_CEILING", "32000")

    class _Silent:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw: object) -> object:
            self.calls += 1
            msg = SimpleNamespace(content="")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    rec = _Silent()
    with pytest.raises(EmptyCompletionError, match="finish_reason=stop"):
        OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=1000, messages=[])
    assert rec.calls == 1


def test_the_sampling_temperature_is_passed_through() -> None:
    """Left unset, an endpoint samples at its own default and every run draws again: the same
    paradigm over the same document returned forty entity types once and nine the next time.
    Every comparison run before this was a single sample of a wide distribution read as a
    measurement."""
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(
        model="m", max_tokens=1000, messages=[], temperature=0.0
    )
    assert rec.seen["temperature"] == 0.0


def test_no_temperature_is_sent_when_none_is_asked_for() -> None:
    """Not every caller wants it pinned, and sending a default we invented would override an
    endpoint's own — silently, and differently per provider."""
    rec = _Rec("ok")
    OpenAICompatClient(client=rec).messages.create(model="m", max_tokens=1000, messages=[])
    assert "temperature" not in rec.seen
