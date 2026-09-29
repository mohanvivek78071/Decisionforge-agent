from decisionforge.llm.base import LLM
from decisionforge.llm.offline import OfflineLLM


def get_llm(provider: str = "offline") -> LLM:
    if provider == "offline":
        return OfflineLLM()
    if provider == "anthropic":
        from decisionforge.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM()
    raise ValueError(f"unknown provider '{provider}' (use 'offline' or 'anthropic')")


__all__ = ["LLM", "OfflineLLM", "get_llm"]
