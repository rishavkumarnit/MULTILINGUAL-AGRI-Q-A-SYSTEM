"""LangGraph state machine for the AI workflow: translate/extract, then reuse,
RAG, or a status message, then translate the answer back."""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from .crop_price import CropPrice, fetch_crop_price
from .models import ChatRequest, ChatResponse
from .rag import RetrievedChunk, find_relevant_chunks, generate_rag_answer
from .semantic_search import SemanticMatch, find_verified_match
from .weather import WeatherForecast, fetch_weather_forecast

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
        "intent": {
            "type": "string",
            "enum": ["weather", "price", "general"],
            "description": (
                "weather if the farmer is asking about weather, rain, or forecast; "
                "price if the farmer is asking about market rate, price, or how much a crop sells for; "
                "general otherwise."
            ),
        },
    },
    "required": ["questionEnglish", "crop", "location", "intent"],
}

EXTRACTION_INPUT_PROMPT = PromptTemplate.from_template("Selected response language: {language}\nQuestion: {question}")
TRANSLATE_BACK_PROMPT = PromptTemplate.from_template(
    "Translate the agricultural answer into {language}. "
    "Preserve all practical details and safety cautions. Return only the translation."
)


class WorkflowState(TypedDict, total=False):
    request: ChatRequest
    question_english: str
    crop: str | None
    location: str | None
    intent: str
    forecast: WeatherForecast | None
    price: CropPrice | None
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
            "only when explicitly stated; never guess either one. "
            "Classify intent as 'weather' if the question asks about weather, rain, temperature, or forecast; "
            "'price' if it asks about market rate, mandi price, or how much a crop sells for; "
            "'general' for everything else. Return only the requested JSON."
        ),
        input=EXTRACTION_INPUT_PROMPT.format(language=LANGUAGE_NAMES.get(request.language, "English"), question=request.question),
        text={"format": {"type": "json_schema", "name": "agri_question_context", "strict": True, "schema": EXTRACTION_SCHEMA}},
    )
    result = json.loads(response.output_text)
    return {
        "question_english": result["questionEnglish"],
        "crop": result.get("crop"),
        "location": result.get("location"),
        "intent": result["intent"],
    }


def _route_after_extraction(state: WorkflowState) -> str:
    if state["intent"] == "weather" and state.get("location"):
        return "fetch_weather"
    if state["intent"] == "price" and state.get("crop"):
        return "fetch_price"
    return "semantic_search"


async def _fetch_weather(state: WorkflowState) -> dict:
    forecast = await fetch_weather_forecast(state["location"])
    if not forecast:
        return {"forecast": None}
    return {
        "forecast": forecast,
        "answer_english": _weather_answer_text(forecast),
        "source": "weather-forecast",
    }


def _route_after_weather(state: WorkflowState) -> str:
    return "translate_back" if state.get("forecast") else "semantic_search"


def _weather_answer_text(forecast: WeatherForecast) -> str:
    lines = [f"5-day forecast for {forecast.location_label}:"]
    for day in forecast.days:
        lines.append(
            f"- {day.date}: {day.description}, {day.temp_min:.0f}-{day.temp_max:.0f}°C, "
            f"{day.precipitation_probability}% chance of rain ({day.precipitation_mm:.1f} mm)"
        )
    return "\n".join(lines)


async def _fetch_price(state: WorkflowState) -> dict:
    price = await fetch_crop_price(state["crop"], state.get("location"))
    if not price:
        return {"price": None}
    return {
        "price": price,
        "answer_english": _price_answer_text(price),
        "source": "price-lookup",
    }


def _route_after_price(state: WorkflowState) -> str:
    return "translate_back" if state.get("price") else "semantic_search"


def _price_answer_text(price: CropPrice) -> str:
    return (
        f"The latest modal price for {price.commodity} at {price.market}, {price.state} "
        f"(as of {price.arrival_date}) is Rs {price.modal_price:.0f} per quintal "
        f"(range Rs {price.min_price:.0f}-{price.max_price:.0f})."
    )


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
        instructions=TRANSLATE_BACK_PROMPT.format(language=LANGUAGE_NAMES.get(request.language, "English")),
        input=answer_english,
    )
    return {"answer_english": response.output_text}


def _build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("translate_extract", _translate_extract)
    builder.add_node("fetch_weather", _fetch_weather)
    builder.add_node("fetch_price", _fetch_price)
    builder.add_node("semantic_search", _semantic_search)
    builder.add_node("rag_retrieve", _rag_retrieve)
    builder.add_node("generate_rag_answer", _generate_rag_answer)
    builder.add_node("status_message", _status_message)
    builder.add_node("translate_back", _translate_back)

    builder.add_edge(START, "translate_extract")
    builder.add_conditional_edges("translate_extract", _route_after_extraction, ["fetch_weather", "fetch_price", "semantic_search"])
    builder.add_conditional_edges("fetch_weather", _route_after_weather, ["translate_back", "semantic_search"])
    builder.add_conditional_edges("fetch_price", _route_after_price, ["translate_back", "semantic_search"])
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
