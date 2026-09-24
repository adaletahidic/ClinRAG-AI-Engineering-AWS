from __future__ import annotations

from app.config import LLM_PROVIDER
from app.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    provider = (LLM_PROVIDER or "groq").lower().strip()

    if provider == "groq":
        from app.llm.groq_provider import GroqProvider
        return GroqProvider()

    if provider == "mistral":
        from app.llm.mistral_provider import MistralProvider
        return MistralProvider()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
    )