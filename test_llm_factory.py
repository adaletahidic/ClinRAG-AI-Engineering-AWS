from __future__ import annotations

from app.llm.factory import LLMProviderFactory


if __name__ == "__main__":
    provider = LLMProviderFactory.create()
    print(type(provider).__name__)

    response = provider.generate(
        system_prompt=(
            "You are a concise AI engineering assistant. "
            "Reply in one sentence."
        ),
        user_prompt="Explain MCP in one sentence.",
    )

    print("\nRESPONSE:")
    print(response)
