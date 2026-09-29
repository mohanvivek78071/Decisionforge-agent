"""Real-LLM provider (Claude). Same interface as the offline provider.

Set ANTHROPIC_API_KEY. Optionally DECISIONFORGE_MODEL (default: claude-sonnet-5-5).
Output is parsed and validated against the pydantic schema, with a repair retry.
"""
from __future__ import annotations

import os
import re

from decisionforge.llm.prompts import SYSTEM, build_prompt


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return text[start : end + 1]


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, max_retries: int = 2):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install 'decisionforge[llm]' to use the anthropic provider") from e
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("DECISIONFORGE_MODEL", "claude-sonnet-5-5")
        self.max_retries = max_retries

    def structured(self, task, payload, model):
        prompt, last_err = build_prompt(task, payload, model), None
        for _ in range(self.max_retries + 1):
            content = prompt + (f"\n\nYour previous reply was invalid ({last_err}). Return valid JSON only." if last_err else "")
            msg = self.client.messages.create(
                model=self.model, max_tokens=4000, system=SYSTEM, messages=[{"role": "user", "content": content}]
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            try:
                return model.model_validate_json(_extract_json(text))
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:400]
        raise RuntimeError(f"LLM failed to produce valid '{task}' output: {last_err}")
