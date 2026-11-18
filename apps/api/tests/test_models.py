import pytest
from pydantic import ValidationError

from main import ChatRequest, ChatResponse
from models import Experience, Profile


def test_chat_message_is_trimmed():
    assert ChatRequest(message="  Hello  ").message == "Hello"


@pytest.mark.parametrize("end_date", ["2026-07", None])
def test_experience_accepts_completed_and_current_roles(end_date):
    experience = Experience(
        id="example", company="Example", title="Engineer",
        startDate="2021-11", endDate=end_date, highlights=["Built an application."],
    )
    assert experience.endDate == end_date


def test_profile_rejects_missing_required_content():
    with pytest.raises(ValidationError):
        Profile.model_validate({"profileId": "default", "name": "Example"})


def test_chat_response_rejects_unsupported_ui_action():
    with pytest.raises(ValidationError):
        ChatResponse(message="Example", action="execute_code")
