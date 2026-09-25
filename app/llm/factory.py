from __future__ import annotations

from app.config import LLM_PROVIDER
from app.llm.base import LLMProvider


class LLMProviderFactory:
    """
    Centralizes provider selection so agents depend on the LLM interface,
    not a concrete provider implementation.
    """

    @staticmethod
    def create() -> LLMProvider:
        provider = (LLM_PROVIDER or "groq").lower().strip()

        if provider == "groq":
            from app.llm.groq_provider import GroqProvider
            return GroqProvider()

        if provider == "mistral":
            from app.llm.mistral_provider import MistralProvider
            return MistralProvider()

        if provider == "bedrock":
            from app.llm.bedrock_provider import BedrockProvider
            return BedrockProvider()

        raise ValueError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
        )


def get_llm_provider() -> LLMProvider:
    """Backward-compatible wrapper around the factory."""
    return LLMProviderFactory.create()