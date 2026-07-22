from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="en", min_length=2, max_length=10)
    conversation_id: str | None = Field(default=None, alias="conversationId")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    answer: str
    # Set by the endpoint layer after process_question() returns; workflow.py doesn't know about conversation IDs.
    conversation_id: str = Field(default="", alias="conversationId")
    question_english: str = Field(alias="questionEnglish")
    # English-language version of `answer`, kept only so conversation history can be
    # replayed as LLM context in one language regardless of what language each turn
    # was actually conducted in. Not meant for display.
    answer_english: str = Field(default="", alias="answerEnglish")
    crop: str | None = None
    location: str | None = None
    similarity: float | None = None
    source: Literal[
        "translation-extraction", "development-fallback", "semantic-reuse", "rag-generated",
        "weather-forecast", "price-lookup", "llm-general",
    ]
    sources: list[str] | None = None

    model_config = {"populate_by_name": True}
