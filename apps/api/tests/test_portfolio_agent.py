from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain.tools import tool
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

from agent_middleware import DailyLimitReached
from models import ChatMessage, ChatResponse, Profile
from portfolio_agent import PortfolioAgent
from quota import Quota
from settings import Settings


def test_answer_passes_history_and_returns_structured_response():
    # Skip construction: this test checks answer(), not model setup.
    agent = PortfolioAgent.__new__(PortfolioAgent)
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = {
        "structured_response": {
            "message": "Here is the resume.",
            "action": "show_resume",
        }
    }

    history = [
        ChatMessage(
            role="assistant",
            content="Would you like the full resume?",
        )
    ]

    response = agent.answer("Yes", history)

    assert response == ChatResponse(
        message="Here is the resume.",
        action="show_resume",
    )

    agent.agent.invoke.assert_called_once_with(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Would you like the full resume?",
                },
                {"role": "user", "content": "Yes"},
            ]
        },
        config={"recursion_limit": 20},
    )


@pytest.mark.parametrize("result", [{}, {"structured_response": None}])
def test_missing_structured_response_is_rejected(result):
    agent = PortfolioAgent.__new__(PortfolioAgent)
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = result
    with pytest.raises(ValueError, match="structured response"):
        agent.answer("Hello")


def test_invalid_structured_action_is_rejected():
    agent = PortfolioAgent.__new__(PortfolioAgent)
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = {
        "structured_response": {"message": "Hello", "action": "execute_code"}
    }
    with pytest.raises(ValidationError):
        agent.answer("Hello")


class ScriptedModel(FakeMessagesListChatModel):
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)


def tool_call(name, args=None, call_id="call"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args or {}, "id": call_id,
    }])


@pytest.fixture
def build_agent(monkeypatch):
    profile = Profile(
        profileId="test", name="Example Engineer", headline="Engineer",
        summary="Builds React applications.", experience=[], skills=[],
        education=[], languages=[],
    )
    settings = Settings(
        _env_file=None, ai_mode="gemini", gemini_api_key="fake-test-key",
        gemini_model="test-model", quota_timezone="America/Los_Angeles",
    )

    def build(responses, daily_limit=10):
        model = ScriptedModel(responses=responses)
        monkeypatch.setattr("portfolio_agent.ChatGoogleGenerativeAI", lambda **kwargs: model)
        quota = Quota(daily_limit, "America/Los_Angeles")
        return PortfolioAgent(settings, profile, quota), model, quota

    return build


@pytest.mark.parametrize("tool_name", ["get_profile", "get_resume"])
def test_real_loop_executes_portfolio_tool(build_agent, monkeypatch, tool_name):
    from agent_tools import create_portfolio_tools

    executed = []

    def tracked_tools(profile):
        wrapped = []
        for original in create_portfolio_tools(profile):
            def track(original=original):
                executed.append(original.name)
                return original.invoke({})
            wrapped.append(tool(original.name, description=original.description)(track))
        return wrapped

    monkeypatch.setattr("portfolio_agent.create_portfolio_tools", tracked_tools)
    action = "show_resume" if tool_name == "get_resume" else None
    agent, model, quota = build_agent([
        tool_call(tool_name),
        tool_call("ChatResponse", {"message": "Verified answer", "action": action}, "final"),
    ])
    response = agent.answer("Show the resume" if action else "What experience?")
    assert response.action == action
    assert executed == [tool_name]
    assert model.calls == quota.used == 2


@pytest.mark.parametrize("limit,expected_calls", [(0, 0), (1, 1)])
def test_daily_quota_stops_real_loop(build_agent, limit, expected_calls):
    agent, model, quota = build_agent([tool_call("get_profile")], daily_limit=limit)
    with pytest.raises(DailyLimitReached):
        agent.answer("What experience?")
    assert model.calls == quota.used == expected_calls


def test_model_call_limit_stops_nonterminating_agent(build_agent, monkeypatch):
    from langchain.agents.middleware import ToolCallLimitMiddleware

    # Raise the other guard only in this test so the model limit is exercised.
    monkeypatch.setattr(
        "portfolio_agent.ToolCallLimitMiddleware",
        lambda **kwargs: ToolCallLimitMiddleware(run_limit=100, exit_behavior="error"),
    )
    # Invalid structured output prompts correction; retries must remain bounded.
    agent, model, quota = build_agent([
        tool_call("ChatResponse", {"message": "Hello", "action": "execute_code"})
    ])
    with pytest.raises(ModelCallLimitExceededError):
        agent.answer("Hello")
    assert model.calls == quota.used == 4


def test_tool_call_limit_stops_repeated_tools(build_agent):
    agent, model, quota = build_agent([tool_call("get_profile")])
    with pytest.raises(ToolCallLimitExceededError):
        agent.answer("What experience?")
    assert model.calls == quota.used == 4
