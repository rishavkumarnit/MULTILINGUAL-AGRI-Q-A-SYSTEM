"""LangGraph state machine for the AI workflow: translate/extract, then reuse,
RAG, or a status message, then translate the answer back."""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from .models import ChatRequest, ChatResponse
from .rag import RetrievedChunk, find_relevant_chunks, generate_rag_answer
from .semantic_search import SemanticMatch, find_verified_match

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


class WorkflowState(TypedDict, total=False):
    request: ChatRequest
    question_english: str
    crop: str | None
    location: str | None
    verified_match: SemanticMatch | None
    chunks: list[RetrievedChunk]
    answer_english: str
    source: str
    sources: list[str] | None
    similarity: float | None


async def _translate_extract(state: WorkflowState) -> dict:
    """Translate to English and extract only explicitly stated context."""
    request = state["request"]
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
    return {
        "question_english": result["questionEnglish"],
        "crop": result.get("crop"),
        "location": result.get("location"),
    }


async def _semantic_search(state: WorkflowState) -> dict:
    match = await find_verified_match(state["question_english"], state.get("crop"), state.get("location"))
    if not match:
        return {"verified_match": None}
    return {
        "verified_match": match,
        "answer_english": match.answer_english,
        "source": "semantic-reuse",
        "similarity": round(match.similarity, 4),
    }


def _route_after_semantic_search(state: WorkflowState) -> str:
    return "translate_back" if state.get("verified_match") else "rag_retrieve"


async def _rag_retrieve(state: WorkflowState) -> dict:
    chunks = await find_relevant_chunks(state["question_english"])
    return {"chunks": chunks}


def _route_after_rag_retrieve(state: WorkflowState) -> str:
    return "generate_rag_answer" if state.get("chunks") else "status_message"


async def _generate_rag_answer(state: WorkflowState) -> dict:
    chunks = state["chunks"]
    answer_english = await generate_rag_answer(state["question_english"], chunks)
    return {
        "answer_english": answer_english,
        "source": "rag-generated",
        "sources": sorted({chunk.title for chunk in chunks}),
    }


async def _status_message(state: WorkflowState) -> dict:
    return {
        "answer_english": _status_message_text(state["question_english"], state.get("crop"), state.get("location")),
        "source": "translation-extraction",
    }


async def _translate_back(state: WorkflowState) -> dict:
    """Reuse trusted English knowledge while respecting the selected response language."""
    request = state["request"]
    answer_english = state["answer_english"]
    if request.language == "en":
        return {"answer_english": answer_english}
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            f"Translate the agricultural answer into {LANGUAGE_NAMES.get(request.language, 'English')}. "
            "Preserve all practical details and safety cautions. Return only the translation."
        ),
        input=answer_english,
    )
    return {"answer_english": response.output_text}


def _build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("translate_extract", _translate_extract)
    builder.add_node("semantic_search", _semantic_search)
    builder.add_node("rag_retrieve", _rag_retrieve)
    builder.add_node("generate_rag_answer", _generate_rag_answer)
    builder.add_node("status_message", _status_message)
    builder.add_node("translate_back", _translate_back)

    builder.add_edge(START, "translate_extract")
    builder.add_edge("translate_extract", "semantic_search")
    builder.add_conditional_edges("semantic_search", _route_after_semantic_search, ["translate_back", "rag_retrieve"])
    builder.add_conditional_edges("rag_retrieve", _route_after_rag_retrieve, ["generate_rag_answer", "status_message"])
    builder.add_edge("generate_rag_answer", "translate_back")
    builder.add_edge("translate_back", END)
    builder.add_edge("status_message", END)
    return builder.compile()


_graph = _build_graph()


async def process_question(request: ChatRequest) -> ChatResponse:
    if not os.getenv("OPENAI_API_KEY"):
        return _development_fallback(request)

    final_state = await _graph.ainvoke({"request": request})
    return ChatResponse(
        answer=final_state["answer_english"],
        questionEnglish=final_state["question_english"],
        crop=final_state.get("crop"),
        location=final_state.get("location"),
        similarity=final_state.get("similarity"),
        source=final_state["source"],
        sources=final_state.get("sources"),
    )


def _development_fallback(request: ChatRequest) -> ChatResponse:
    """Keeps local development usable without silently claiming translation occurred."""
    crop = next((item for item in KNOWN_CROPS if re.search(rf"\b{re.escape(item)}\b", request.question, re.IGNORECASE)), None)
    return ChatResponse(
        answer=(
            "Translation and extraction are ready, but OPENAI_API_KEY is not configured. "
            "Showing a local-development fallback only.\n\n"
            + _status_message_text(request.question, crop, None)
        ),
        questionEnglish=request.question,
        crop=crop,
        location=None,
        source="development-fallback",
    )


def _status_message_text(question_english: str, crop: str | None, location: str | None) -> str:
    details = [f"English question: {question_english}", f"Crop: {crop or 'not provided'}", f"Location: {location or 'not provided'}"]
    return "\n".join(details)
