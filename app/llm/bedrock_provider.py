from __future__ import annotations

import json

from app.config import BEDROCK_MODEL
from app.llm.base import LLMProvider


class BedrockProvider(LLMProvider):
    """AWS Bedrock adapter kept behind the existing provider contract."""

    def __init__(self, client=None, model: str | None = None):
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for the Bedrock provider."
                ) from exc
            client = boto3.client("bedrock-runtime")

        self.client = client
        self.model = model or BEDROCK_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
        )
        content = response.get("output", {}).get("message", {}).get(
            "content", []
        )
        answer = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )
        if not answer:
            raise RuntimeError("Bedrock returned an empty response.")
        return answer
