from unittest.mock import MagicMock

import pytest

from api.agents.tools import create_portfolio_tools
from api.core.paths import PROFILE_PATH
from api.schemas.profile import Profile
from api.rag.knowledge import KnowledgeChunk
from api.rag.retrieval import KnowledgeIndex, SearchResult


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


def test_search_tool_is_absent_without_index():
    profile = Profile.model_validate_json(PROFILE_PATH.read_text(encoding="utf-8"))
    assert {tool.name for tool in create_portfolio_tools(profile)} == {
        "get_profile", "get_resume",
    }


def test_search_tool_returns_evidence_and_citation_metadata():
    profile = Profile.model_validate_json(PROFILE_PATH.read_text(encoding="utf-8"))
    chunk = KnowledgeChunk(
        id="python", profile_id=profile.profileId, source_id="experience:intuit",
        document_version="v1", text="Python pipeline experience at Intuit.",
    )
    index = MagicMock(spec=KnowledgeIndex)
    index.search.return_value = [SearchResult(chunk=chunk, score=0.75)]
    tools = {tool.name: tool for tool in create_portfolio_tools(profile, index)}
    result = tools["search_knowledge"].invoke({"query": "Python experience"})
    assert result == [{
        "chunkId": "python", "sourceId": "experience:intuit",
        "documentVersion": "v1", "text": chunk.text, "score": 0.75,
    }]
    index.search.assert_called_once_with("Python experience", top_k=3)


def test_search_tool_returns_empty_results():
    profile = Profile.model_validate_json(PROFILE_PATH.read_text(encoding="utf-8"))
    index = MagicMock(spec=KnowledgeIndex)
    index.search.return_value = []
    tools = {tool.name: tool for tool in create_portfolio_tools(profile, index)}
    assert tools["search_knowledge"].invoke({"query": "Unknown experience"}) == []


def test_search_tool_validates_missing_query_before_search():
    from pydantic import ValidationError
    profile = Profile.model_validate_json(PROFILE_PATH.read_text(encoding="utf-8"))
    index = MagicMock(spec=KnowledgeIndex)
    tools = {tool.name: tool for tool in create_portfolio_tools(profile, index)}
    with pytest.raises(ValidationError):
        tools["search_knowledge"].invoke({})
    index.search.assert_not_called()
