from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=200)
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)

class Citation(BaseModel):
    chunkId: str
    sourceId: str
    documentVersion: str
    title: str
    text: str


class ChatResponse(BaseModel):
    message: str
    action: Literal['show_resume'] | None = None
    citations: list[Citation] = Field(default_factory=list, max_length=3)
