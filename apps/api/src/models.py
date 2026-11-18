from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class SkillGroup(BaseModel):
    category: str
    items: list[str]


class Education(BaseModel):
    degree: str
    institution: str


class Language(BaseModel):
    name: str
    proficiency: str


class Experience(BaseModel):
    id: str
    company: str
    title: str
    startDate: str
    endDate: str | None
    highlights: list[str]

class Profile(BaseModel):
    profileId: str
    name: str
    headline: str
    summary: str
    experience: list[Experience]
    skills: list[SkillGroup]
    education: list[Education]
    languages: list[Language]
    
class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=200)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=10,
    )

class ChatResponse(BaseModel):
    message: str
    action: Literal["show_resume"] | None = None