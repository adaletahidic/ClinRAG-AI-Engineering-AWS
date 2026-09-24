from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

#LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

#MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
#MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")