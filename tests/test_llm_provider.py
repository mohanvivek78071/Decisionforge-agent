"""The Claude provider can't be called in CI, but its parsing/retry contract can be tested with a stub."""
from types import SimpleNamespace

import pytest

from decisionforge.llm.anthropic_client import AnthropicLLM, _extract_json
from decisionforge.schemas import NextStep


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def make(replies):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model, llm.max_retries = "stub", 2
    llm.client = SimpleNamespace(messages=FakeMessages(replies))
    return llm


def test_extracts_json_from_fenced_reply():
    assert _extract_json('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'


def test_retries_on_invalid_then_succeeds_and_untrusted_data_is_tagged():
    llm = make(["not json at all", '{"action": "stop", "reason": "done"}'])
    out = llm.structured("next_step", {"question": "q", "ledger": []}, NextStep)
    assert out.action == "stop"
    calls = llm.client.messages.calls
    assert len(calls) == 2 and "previous reply was invalid" in calls[1]["messages"][0]["content"]
    assert "<data>" in calls[0]["messages"][0]["content"] and "untrusted" in calls[0]["system"]


def test_gives_up_after_retries():
    with pytest.raises(RuntimeError):
        make(["nope", "nope", "nope"]).structured("next_step", {"question": "q", "ledger": []}, NextStep)
