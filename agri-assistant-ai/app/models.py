from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="en", min_length=2, max_length=10)
    conversation_id: str | None = Field(default=None, alias="conversationId")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    answer: str
    question_english: str = Field(alias="questionEnglish")
    crop: str | None = None
    location: str | None = None
    similarity: float | None = None
    source: Literal["translation-extraction", "development-fallback", "semantic-reuse", "rag-generated", "weather-forecast"]
    sources: list[str] | None = None

    model_config = {"populate_by_name": True}
