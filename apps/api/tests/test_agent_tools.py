import json

from agent_tools import create_portfolio_tools
from main import PROFILE_PATH
from models import Profile


def test_profile_tool_returns_verified_data():
    profile = Profile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    tools = {
        tool.name: tool
        for tool in create_portfolio_tools(profile)
    }

    result = tools["get_profile"].invoke({})

    assert result == profile.model_dump()


def test_resume_tool_requests_resume_view():
    profile = Profile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    tools = {
        tool.name: tool
        for tool in create_portfolio_tools(profile)
    }

    result = tools["get_resume"].invoke({})

    assert result == {
        "profileId": profile.profileId,
        "action": "show_resume",
    }