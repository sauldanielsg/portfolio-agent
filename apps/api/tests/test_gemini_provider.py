import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx2
from google.genai import errors

from gemini_provider import GeminiProvider, ProviderError
from models import ChatMessage, ChatResponse, Profile
from settings import Settings


@pytest.fixture
def provider_and_client(monkeypatch):
    settings = Settings(
        _env_file=None,
        ai_mode="gemini",
        gemini_api_key="fake-test-key",
        gemini_model="test-model",
        max_output_tokens=500,
        quota_timezone="America/Los_Angeles",
    )
    client = MagicMock()
    client_class = MagicMock()
    client_class.return_value.__enter__.return_value = client
    monkeypatch.setattr("gemini_provider.genai.Client", client_class)
    return GeminiProvider(settings), client


@pytest.fixture
def profile():
    return Profile(
        profileId="test",
        name="Example Engineer",
        headline="Full-Stack Engineer",
        summary="Builds React applications.",
        experience=[],
        skills=[],
        education=[],
        languages=[],
    )


def test_provider_returns_validated_response(provider_and_client, profile):
    provider, client = provider_and_client
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"message": "They build React applications.", "action": null}'
    )
    question = "What frontend experience do they have?"
    result = provider.answer(question, profile)

    assert result.message == "They build React applications."
    assert result.action is None
    client.models.generate_content.assert_called_once()
    arguments = client.models.generate_content.call_args.kwargs
    assert arguments["model"] == "test-model"
    assert arguments["config"].max_output_tokens == 500
    assert arguments["config"].response_schema is ChatResponse
    assert json.loads(arguments["contents"]) == {
        "profile": profile.model_dump(),
        "conversation_history": [],
        "visitor_question": question,
    }


def test_provider_includes_prior_messages_in_order(provider_and_client, profile):
    provider, client = provider_and_client
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"message": "Here is the resume.", "action": "show_resume"}'
    )
    history = [
        ChatMessage(role="user", content="What experience does Saul have?"),
        ChatMessage(role="assistant", content="Would you like the full resume?"),
    ]
    result = provider.answer("Yes", profile, history)
    payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])
    assert payload["conversation_history"] == [entry.model_dump() for entry in history]
    assert payload["visitor_question"] == "Yes"
    assert result.action == "show_resume"


@pytest.mark.parametrize("text", [None, ""])
def test_empty_response_is_rejected(provider_and_client, profile, text):
    provider, client = provider_and_client
    client.models.generate_content.return_value = SimpleNamespace(text=text)
    with pytest.raises(ProviderError, match="empty response"):
        provider.answer("Hello", profile)


@pytest.mark.parametrize("text", [
    "not valid JSON",
    '{"action": null}',
    '{"message": "Hello", "action": "execute_code"}',
])
def test_invalid_response_is_rejected(provider_and_client, profile, text):
    provider, client = provider_and_client
    client.models.generate_content.return_value = SimpleNamespace(text=text)
    with pytest.raises(ProviderError, match="invalid response"):
        provider.answer("Hello", profile)


@pytest.mark.parametrize("code", [500, 503])
def test_api_errors_are_wrapped(provider_and_client, profile, code):
    provider, client = provider_and_client
    original = errors.APIError(code, {"error": {"message": "private provider detail"}})
    client.models.generate_content.side_effect = original
    with pytest.raises(ProviderError, match="temporarily unavailable") as caught:
        provider.answer("Hello", profile)
    assert caught.value.__cause__ is original
    assert "private provider detail" not in str(caught.value)
    client.models.generate_content.assert_called_once()


def test_provider_rate_limit_is_distinguished(provider_and_client, profile):
    from gemini_provider import ProviderRateLimitError
    provider, client = provider_and_client
    client.models.generate_content.side_effect = errors.APIError(429, {})
    with pytest.raises(ProviderRateLimitError):
        provider.answer("Hello", profile)


@pytest.mark.parametrize("error_type", [httpx2.ConnectError, httpx2.ReadTimeout])
def test_transport_errors_are_wrapped(provider_and_client, profile, error_type):
    provider, client = provider_and_client
    original = error_type("private connection detail")
    client.models.generate_content.side_effect = original
    with pytest.raises(ProviderError, match="Could not connect") as caught:
        provider.answer("Hello", profile)
    assert caught.value.__cause__ is original
    assert "private connection detail" not in str(caught.value)
    client.models.generate_content.assert_called_once()
