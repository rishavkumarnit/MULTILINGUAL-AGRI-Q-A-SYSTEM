"""Translation and context-extraction stage of the AI workflow."""

import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .models import ChatRequest, ChatResponse
from .rag import find_relevant_chunks, generate_rag_answer
from .semantic_search import find_verified_match

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi"}
KNOWN_CROPS = ("wheat", "rice", "paddy", "mustard", "maize", "corn", "cotton", "potato", "tomato", "sugarcane", "soybean", "chickpea")

load_dotenv()

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questionEnglish": {"type": "string", "description": "Faithful English translation of the farmer's question."},
        "crop": {"type": ["string", "null"], "description": "Crop explicitly mentioned, or null."},
        "location": {"type": ["string", "null"], "description": "District, village, state, or location explicitly mentioned, or null."},
    },
    "required": ["questionEnglish", "crop", "location"],
}


async def process_question(request: ChatRequest) -> ChatResponse:
    """Translate to English and extract only explicitly stated context."""
    if not os.getenv("OPENAI_API_KEY"):
        return _development_fallback(request)

    client = AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            "You are the preprocessing stage for an agricultural assistant. "
            "Translate the farmer's question faithfully into English. Extract a crop and location "
            "only when explicitly stated; never guess either one. Return only the requested JSON."
        ),
        input=f"Selected response language: {LANGUAGE_NAMES.get(request.language, 'English')}\nQuestion: {request.question}",
        text={"format": {"type": "json_schema", "name": "agri_question_context", "strict": True, "schema": EXTRACTION_SCHEMA}},
    )
    result = json.loads(response.output_text)
    match = await find_verified_match(result["questionEnglish"], result.get("crop"), result.get("location"))
    if match:
        return ChatResponse(
            answer=await _translate_answer_for_user(match.answer_english, request.language, client),
            questionEnglish=result["questionEnglish"],
            crop=result.get("crop"),
            location=result.get("location"),
            similarity=round(match.similarity, 4),
            source="semantic-reuse",
        )
    chunks = await find_relevant_chunks(result["questionEnglish"])
    if chunks:
        answer_english = await generate_rag_answer(result["questionEnglish"], chunks, client)
        return ChatResponse(
            answer=await _translate_answer_for_user(answer_english, request.language, client),
            questionEnglish=result["questionEnglish"],
            crop=result.get("crop"),
            location=result.get("location"),
            source="rag-generated",
            sources=sorted({chunk.title for chunk in chunks}),
        )
    return ChatResponse(
        answer=_status_message(result["questionEnglish"], result.get("crop"), result.get("location")),
        questionEnglish=result["questionEnglish"],
        crop=result.get("crop"),
        location=result.get("location"),
        source="translation-extraction",
    )


async def _translate_answer_for_user(answer_english: str, language: str, client: AsyncOpenAI) -> str:
    """Reuse trusted English knowledge while respecting the selected response language."""
    if language == "en":
        return answer_english
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            f"Translate the agricultural answer into {LANGUAGE_NAMES.get(language, 'English')}. "
            "Preserve all practical details and safety cautions. Return only the translation."
        ),
        input=answer_english,
    )
    return response.output_text


def _development_fallback(request: ChatRequest) -> ChatResponse:
    """Keeps local development usable without silently claiming translation occurred."""
    crop = next((item for item in KNOWN_CROPS if re.search(rf"\b{re.escape(item)}\b", request.question, re.IGNORECASE)), None)
    return ChatResponse(
        answer=(
            "Translation and extraction are ready, but OPENAI_API_KEY is not configured. "
            "Showing a local-development fallback only.\n\n"
            + _status_message(request.question, crop, None)
        ),
        questionEnglish=request.question,
        crop=crop,
        location=None,
        source="development-fallback",
    )


def _status_message(question_english: str, crop: str | None, location: str | None) -> str:
    details = [f"English question: {question_english}", f"Crop: {crop or 'not provided'}", f"Location: {location or 'not provided'}"]
    return "\n".join(details)
