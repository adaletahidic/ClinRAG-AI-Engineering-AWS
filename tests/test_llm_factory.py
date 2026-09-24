from __future__ import annotations

from app.llm.factory import LLMProviderFactory


def test_llm_factory_creates_configured_provider():
    provider = LLMProviderFactory.create()

    assert provider is not None
    assert provider.__class__.__name__ in {"GroqProvider", "MistralProvider"}
