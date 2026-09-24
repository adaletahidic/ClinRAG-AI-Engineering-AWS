from __future__ import annotations


class FakeLLM:
    """
    Deterministic fake LLM used by unit tests.

    It never calls an external API.
    """

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        return self.response