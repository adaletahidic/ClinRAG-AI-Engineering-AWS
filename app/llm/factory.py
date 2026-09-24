from __future__ import annotations

from app.config import LLM_PROVIDER
from app.llm.base import LLMProvider
from app.llm.mistral_provider import MistralProvider


def get_llm_provider() -> LLMProvider:
    provider = LLM_PROVIDER.lower().strip()

    if provider == "mistral":
        return MistralProvider()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
    )