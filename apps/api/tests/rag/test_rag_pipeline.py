import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from api.agents.middleware import DailyLimitReached
from api.rag.knowledge import KnowledgeChunk
from api.schemas.profile import Profile
from api.schemas.chat import ChatMessage
from api.agents.portfolio import PortfolioAgent
from api.usage.quota import Quota
from api.rag.pipeline import GroundedAnswer, RequestPlan, SingleRetrievalRAG
from api.rag.retrieval import KnowledgeIndex, SearchResult
from api.core.settings import Settings


@pytest.fixture
def workflow():
    profile = Profile(
        profileId="test", name="Example Engineer", headline="Engineer",
        summary="A general summary that is not retrieved evidence.",
        experience=[], skills=[], education=[], languages=[],
    )
    model = MagicMock()
    planner, writer = MagicMock(), MagicMock()
    model.with_structured_output.side_effect = [planner, writer]
    planner.invoke.return_value = RequestPlan(intent="knowledge")
    writer.invoke.return_value = GroundedAnswer(message="Built Python pipelines at Example Co.", cited_chunk_ids=["python"])
    index = MagicMock(spec=KnowledgeIndex)
    index.search.return_value = [SearchResult(
        chunk=KnowledgeChunk(
            id="python", profile_id="test", source_id="experience:example",
            document_version="v1", text="Built Python pipelines at Example Co.",
        ), score=0.8,
    )]
    quota = Quota(10, "America/Los_Angeles")
    pipeline = SingleRetrievalRAG(model, profile, index, quota)
    return pipeline, model, planner, writer, index, quota


def payload(runnable):
    return json.loads(runnable.invoke.call_args.args[0][1][1])


def test_one_search_then_one_grounded_answer(workflow):
    pipeline, model, planner, writer, index, quota = workflow
    response = pipeline.answer("What Python experience?")
    index.search.assert_called_once_with("What Python experience?", top_k=3)
    planner.invoke.assert_called_once()
    writer.invoke.assert_called_once()
    assert quota.used == 2
    assert response.message == "Built Python pipelines at Example Co."
    assert response.action is None
    assert [citation.model_dump() for citation in response.citations] == [{
        "chunkId": "python", "sourceId": "experience:example", "documentVersion": "v1",
        "title": "Professional experience", "text": "Built Python pipelines at Example Co.",
    }]
    evidence = payload(writer)["verified_evidence"]
    assert evidence == [{
        "chunkId": "python", "sourceId": "experience:example",
        "documentVersion": "v1", "text": "Built Python pipelines at Example Co.",
    }]
    assert "general summary" not in json.dumps(payload(writer))
    assert model.with_structured_output.call_args_list[0].args == (RequestPlan,)
    assert model.with_structured_output.call_args_list[1].args == (GroundedAnswer,)
    assert all(call.kwargs == {"method": "json_schema"} for call in model.with_structured_output.call_args_list)


def test_empty_search_returns_normal_answer_without_generation_or_second_search(workflow):
    pipeline, _, _, writer, index, quota = workflow
    index.search.return_value = []
    response = pipeline.answer("Does Saul have Kubernetes experience?")
    index.search.assert_called_once_with("Does Saul have Kubernetes experience?", top_k=3)
    writer.invoke.assert_not_called()
    assert "couldn't find matching information documented" in response.message
    assert "Docker" not in response.message
    assert response.action is None
    assert response.citations == []
    assert quota.used == 1  # The intent/context plan still uses one model call.


def test_self_contained_question_cannot_be_broadened_by_planner(workflow):
    pipeline, _, planner, _, index, _ = workflow
    planner.invoke.return_value = RequestPlan(intent="knowledge", query="container orchestration Docker")
    history = [ChatMessage(role="user", content="What Python experience?")]
    pipeline.answer("Kubernetes experience?", history)
    index.search.assert_called_once_with("Kubernetes experience?", top_k=3)


def test_followup_uses_resolved_question_and_passes_history(workflow):
    pipeline, _, planner, writer, index, quota = workflow
    history = [
        ChatMessage(role="user", content="What Python experience?"),
        ChatMessage(role="assistant", content="Python pipelines are documented."),
    ]
    planner.invoke.return_value = RequestPlan(
        intent="knowledge", needs_context=True, query="At which company did the owner build Python pipelines?",
    )
    pipeline.answer("At which company?", history)
    index.search.assert_called_once_with(
        "At which company did the owner build Python pipelines?", top_k=3,
    )
    assert payload(planner)["history"] == [message.model_dump() for message in history]
    assert payload(writer)["history"] == [message.model_dump() for message in history]
    assert quota.used == 2


def test_no_prior_visitor_question_means_original_query_is_preserved(workflow):
    pipeline, _, planner, _, index, _ = workflow
    planner.invoke.return_value = RequestPlan(intent="knowledge", needs_context=True, query="Docker")
    pipeline.answer("Kubernetes experience?", [ChatMessage(role="assistant", content="Welcome")])
    index.search.assert_called_once_with("Kubernetes experience?", top_k=3)


@pytest.mark.parametrize("message", ["Show the resume", "Yes, show it"])
def test_resume_uses_tool_without_search_or_answer_generation(workflow, message):
    pipeline, _, planner, writer, index, quota = workflow
    planner.invoke.return_value = RequestPlan(intent="resume")
    pipeline.resume_tool = MagicMock(wraps=pipeline.resume_tool)
    history = [ChatMessage(role="assistant", content="Would you like the resume?")]
    response = pipeline.answer(message, history)
    pipeline.resume_tool.invoke.assert_called_once_with({})
    assert response.action == "show_resume"
    assert response.citations == []
    assert "Example Engineer" in response.message
    index.search.assert_not_called()
    writer.invoke.assert_not_called()
    assert quota.used == 1


def test_conversation_does_not_search(workflow):
    pipeline, _, planner, writer, index, quota = workflow
    planner.invoke.return_value = RequestPlan(intent="conversation")
    assert pipeline.answer("Hello").action is None
    index.search.assert_not_called()
    writer.invoke.assert_not_called()
    assert quota.used == 1


def test_generated_answer_cannot_trigger_resume_ui(workflow):
    pipeline, _, _, writer, _, _ = workflow
    writer.invoke.return_value = {"message": "An answer", "action": "show_resume", "cited_chunk_ids": ["python"]}
    assert pipeline.answer("Python experience?").action is None


def test_exhausted_daily_quota_blocks_plan_and_search(workflow):
    pipeline, _, planner, writer, index, quota = workflow
    quota.daily_limit = 0
    with pytest.raises(DailyLimitReached):
        pipeline.answer("Python experience?")
    planner.invoke.assert_not_called()
    writer.invoke.assert_not_called()
    index.search.assert_not_called()


def test_quota_is_checked_again_before_answer_generation(workflow):
    pipeline, _, _, writer, index, quota = workflow
    quota.daily_limit = 1
    with pytest.raises(DailyLimitReached):
        pipeline.answer("Python experience?")
    assert quota.used == 1
    index.search.assert_called_once()
    writer.invoke.assert_not_called()


def test_failed_generation_is_not_retried_and_consumes_reservation(workflow):
    pipeline, _, _, writer, index, quota = workflow
    writer.invoke.side_effect = RuntimeError("provider failed")
    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.answer("Python experience?")
    writer.invoke.assert_called_once()
    index.search.assert_called_once()
    assert quota.used == 2


def test_invalid_plan_stops_without_search_or_generation(workflow):
    pipeline, _, planner, writer, index, quota = workflow
    planner.invoke.return_value = {"intent": "search_forever"}
    with pytest.raises(ValidationError):
        pipeline.answer("Python experience?")
    index.search.assert_not_called()
    writer.invoke.assert_not_called()
    assert quota.used == 1


def test_each_request_has_its_own_single_search(workflow):
    pipeline, _, planner, writer, index, quota = workflow
    for _ in range(3):
        pipeline.answer("Python experience?")
    assert planner.invoke.call_count == writer.invoke.call_count == index.search.call_count == 3
    assert quota.used == 6


def test_rag_mode_does_not_construct_or_invoke_an_agent_loop(workflow, monkeypatch):
    pipeline, model, _, _, index, quota = workflow
    # A new pipeline needs its own pair of structured runnables.
    model.with_structured_output.side_effect = [pipeline.planner, pipeline.writer]
    monkeypatch.setattr("api.agents.portfolio.ChatGoogleGenerativeAI", lambda **kwargs: model)
    create_agent = MagicMock(side_effect=AssertionError("RAG must not build a graph"))
    monkeypatch.setattr("api.agents.portfolio.create_agent", create_agent)
    settings = Settings(_env_file=None, ai_mode="gemini", gemini_api_key="fake", gemini_model="test")
    agent = PortfolioAgent(settings, pipeline.profile, quota, index)
    response = agent.answer("Python experience?")
    assert response.message == "Built Python pipelines at Example Co."
    create_agent.assert_not_called()
    index.search.assert_called_once()


@pytest.mark.parametrize("ids", [["invented"], ["python", "old-profile-chunk"]])
def test_invalid_citations_suppress_the_generated_answer(workflow, ids):
    pipeline, _, _, writer, _, quota = workflow
    writer.invoke.return_value = GroundedAnswer(message="Unsupported claim must not be displayed.", cited_chunk_ids=ids)
    response = pipeline.answer("Python experience?")
    assert response.citations == []
    assert "couldn't verify the sources" in response.message
    assert "Unsupported claim" not in response.message
    assert quota.used == 2  # No repair loop or retry.


def test_missing_citations_return_insufficient_evidence_without_generated_claims(workflow):
    pipeline, _, _, writer, _, _ = workflow
    writer.invoke.return_value = GroundedAnswer(message="An uncited claim.")
    response = pipeline.answer("Python experience?")
    assert "couldn't find enough relevant evidence" in response.message
    assert "uncited claim" not in response.message
    assert response.citations == []


def test_duplicate_citations_are_deduplicated(workflow):
    pipeline, _, _, writer, _, _ = workflow
    writer.invoke.return_value = GroundedAnswer(message="Python pipelines.", cited_chunk_ids=["python", "python"])
    assert len(pipeline.answer("Python experience?").citations) == 1


def test_history_cannot_authorize_a_citation_not_retrieved_now(workflow):
    pipeline, _, _, writer, _, _ = workflow
    writer.invoke.return_value = GroundedAnswer(message="A stale answer.", cited_chunk_ids=["previous-source"])
    response = pipeline.answer("At which company?", [
        ChatMessage(role="assistant", content="Source previous-source from an earlier answer."),
    ])
    assert response.citations == []
    assert "stale answer" not in response.message


def test_source_label_and_evidence_come_from_profile_and_retrieved_chunk(workflow):
    from api.schemas.profile import Experience

    pipeline, _, _, writer, index, _ = workflow
    pipeline.profile.experience = [Experience(
        id="example", company="Example Co", title="Engineer", startDate="2024-01",
        endDate=None, highlights=["Built Python pipelines."],
    )]
    writer.invoke.return_value = {
        "message": "Python pipelines.", "cited_chunk_ids": ["python"],
        "title": "Forged title", "text": "Forged evidence", "documentVersion": "fake",
    }
    citation = pipeline.answer("Python experience?").citations[0]
    assert citation.title == "Example Co — Engineer"
    assert citation.text == index.search.return_value[0].chunk.text
    assert citation.documentVersion == "v1"
