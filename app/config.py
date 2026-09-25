from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BEDROCK_MODEL = os.getenv(
    "BEDROCK_MODEL",
    "amazon.nova-lite-v1:0",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
WORKFLOW_TIMEOUT_SECONDS = float(
    os.getenv("WORKFLOW_TIMEOUT_SECONDS", "120")
)
MAX_REQUEST_BODY_BYTES = int(
    os.getenv("MAX_REQUEST_BODY_BYTES", "1048576")
)