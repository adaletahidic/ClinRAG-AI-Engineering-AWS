from __future__ import annotations

import time

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

from app.config import MISTRAL_API_KEY, MISTRAL_MODEL
from app.llm.base import LLMProvider


class MistralProvider(LLMProvider):

    def __init__(self):
        if Mistral is None:
            raise RuntimeError(
                "mistralai is not installed. Install it with: pip install mistralai"
            )
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is not configured.")

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MISTRAL_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )

                return response.choices[0].message.content or ""

            except Exception as exc:
                if "429" not in str(exc):
                    raise

                if attempt == max_retries - 1:
                    raise

                wait_seconds = 2 ** attempt
                time.sleep(wait_seconds)

        raise RuntimeError("Mistral request failed.")