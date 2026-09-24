from unittest.mock import MagicMock, patch
import pytest

from app.llm.groq_provider import GroqProvider


def test_groq_provider_initialization_missing_key():
    with patch("app.llm.groq_provider.GROQ_API_KEY", None):
        with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
            GroqProvider()


def test_groq_provider_initialization_success():
    with patch("app.llm.groq_provider.GROQ_API_KEY", "mock_key"):
        with patch("app.llm.groq_provider.Groq") as mock_groq_cls:
            provider = GroqProvider()
            assert provider.model == "openai/gpt-oss-20b"
            mock_groq_cls.assert_called_once_with(api_key="mock_key")


def test_groq_provider_generate_success():
    with patch("app.llm.groq_provider.GROQ_API_KEY", "mock_key"):
        with patch("app.llm.groq_provider.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client

            mock_choice = MagicMock()
            mock_choice.message.content = "Retrieval-Augmented Generation."
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            provider = GroqProvider()
            result = provider.generate(
                system_prompt="You are an assistant.",
                user_prompt="What is RAG?",
            )
            assert result == "Retrieval-Augmented Generation."
            mock_client.chat.completions.create.assert_called_once()


def test_groq_provider_empty_response():
    with patch("app.llm.groq_provider.GROQ_API_KEY", "mock_key"):
        with patch("app.llm.groq_provider.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client

            mock_choice = MagicMock()
            mock_choice.message.content = ""
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            provider = GroqProvider()
            with pytest.raises(RuntimeError, match="Groq returned an empty response"):
                provider.generate(
                    system_prompt="system",
                    user_prompt="user",
                )