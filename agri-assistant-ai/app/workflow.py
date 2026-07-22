"""LangGraph state machine for the AI workflow: translate/extract, then verified-answer
reuse, then an agent that decides which tools (if any) to call. Translation back into
the requested language happens outside the graph (see translate_answer/
translate_answer_stream) so the streaming endpoint can token-stream it."""

import json
import os
import re
from collections.abc import AsyncIterator
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from .agent import run_agent
from .conversations import CONTEXT_TURN_LIMIT
from .models import ChatRequest, ChatResponse
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

EXTRACTION_INPUT_PROMPT = PromptTemplate.from_template("{history}Selected response language: {language}\nQuestion: {question}")
TRANSLATE_BACK_PROMPT = PromptTemplate.from_template(
    "Translate the agricultural answer into {language}. "
    "Preserve all practical details and safety cautions. Return only the translation."
)


class WorkflowState(TypedDict, total=False):
    request: ChatRequest
    history: str
    question_english: str
    crop: str | None
    location: str | None
    verified_match: SemanticMatch | None
    answer_english: str
    source: str
    sources: list[str] | None
    similarity: float | None


def _format_history(messages: list[dict] | None) -> str:
    """Builds English-only conversation context from persisted assistant messages —
    each already carries both sides of its turn (questionEnglish/answerEnglish), so no
    separate user-message lookup or pairing is needed. Capped to the most recent
    CONTEXT_TURN_LIMIT turns regardless of how much history is actually stored."""
    if not messages:
        return ""
    turns = [m for m in messages if m.get("questionEnglish") and m.get("answerEnglish")]
    recent = turns[-CONTEXT_TURN_LIMIT:]
    return "\n".join(f"Farmer: {turn['questionEnglish']}\nAssistant: {turn['answerEnglish']}" for turn in recent)


async def _translate_extract(state: WorkflowState) -> dict:
    """Translate to English and extract only explicitly stated context."""
    request = state["request"]
    history = state.get("history", "")
    history_block = f"Conversation so far (for resolving follow-ups/pronouns only):\n{history}\n\n" if history else ""
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            "You are the preprocessing stage for an agricultural assistant. "
            "Translate the farmer's question faithfully into English. Extract a crop and location "
            "only when explicitly stated in the current question, or clearly implied by the immediately "
            "preceding conversation turn (e.g. a follow-up like 'and for rice?'); never guess either one. "
            "Return only the requested JSON."
        ),
        input=EXTRACTION_INPUT_PROMPT.format(
            history=history_block, language=LANGUAGE_NAMES.get(request.language, "English"), question=request.question
        ),
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
    return "end" if state.get("verified_match") else "agent"


async def _agent(state: WorkflowState) -> dict:
    result = await run_agent(
        state["question_english"], state.get("crop"), state.get("location"), state.get("history", "")
    )
    if result.source == "ungrounded":
        return {
            "answer_english": _status_message_text(state["question_english"], state.get("crop"), state.get("location")),
            "source": "translation-extraction",
        }
    return {"answer_english": result.answer_english, "source": result.source, "sources": result.sources}


def _build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("translate_extract", _translate_extract)
    builder.add_node("semantic_search", _semantic_search)
    builder.add_node("agent", _agent)

    builder.add_edge(START, "translate_extract")
    builder.add_edge("translate_extract", "semantic_search")
    builder.add_conditional_edges("semantic_search", _route_after_semantic_search, {"end": END, "agent": "agent"})
    builder.add_edge("agent", END)
    return builder.compile()


_graph = _build_graph()


async def translate_answer(answer_english: str, language: str) -> str:
    """Reuse trusted English knowledge while respecting the selected response language."""
    if language == "en":
        return answer_english
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=TRANSLATE_BACK_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English")),
        input=answer_english,
    )
    return response.output_text


async def translate_answer_stream(answer_english: str, language: str) -> AsyncIterator[str]:
    """Same as translate_answer, but yields text deltas as they arrive for language != 'en'."""
    if language == "en":
        yield answer_english
        return
    client = AsyncOpenAI()
    stream = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=TRANSLATE_BACK_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English")),
        input=answer_english,
        stream=True,
    )
    async for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


async def process_question(request: ChatRequest, history: list[dict] | None = None) -> ChatResponse:
    if not os.getenv("OPENAI_API_KEY"):
        return _development_fallback(request)

    final_state = await _graph.ainvoke({"request": request, "history": _format_history(history)})
    answer_english = final_state["answer_english"]
    source = final_state["source"]
    answer = answer_english if source == "translation-extraction" else await translate_answer(answer_english, request.language)
    return ChatResponse(
        answer=answer,
        answerEnglish=answer_english,
        questionEnglish=final_state["question_english"],
        crop=final_state.get("crop"),
        location=final_state.get("location"),
        similarity=final_state.get("similarity"),
        source=source,
        sources=final_state.get("sources"),
    )


async def stream_question(request: ChatRequest, history: list[dict] | None = None) -> AsyncIterator[dict]:
    """Same pipeline as process_question, but yields {"type": "metadata"|"delta"|"done", ...}
    dicts so the caller can stream the final answer text as it's generated. The
    multi-step pipeline up to the English answer is not streamed, only the final
    presentation/translation step is (or, for English, sent as a single delta)."""
    if not os.getenv("OPENAI_API_KEY"):
        fallback = _development_fallback(request)
        yield {
            "type": "metadata",
            "questionEnglish": fallback.question_english,
            "answerEnglish": fallback.answer_english,
            "crop": fallback.crop,
            "location": fallback.location,
            "similarity": fallback.similarity,
            "source": fallback.source,
            "sources": fallback.sources,
        }
        yield {"type": "delta", "text": fallback.answer}
        yield {"type": "done"}
        return

    final_state = await _graph.ainvoke({"request": request, "history": _format_history(history)})
    answer_english = final_state["answer_english"]
    source = final_state["source"]
    yield {
        "type": "metadata",
        "questionEnglish": final_state["question_english"],
        "answerEnglish": answer_english,
        "crop": final_state.get("crop"),
        "location": final_state.get("location"),
        "similarity": final_state.get("similarity"),
        "source": source,
        "sources": final_state.get("sources"),
    }

    if source == "translation-extraction" or request.language == "en":
        yield {"type": "delta", "text": answer_english}
    else:
        async for chunk in translate_answer_stream(answer_english, request.language):
            yield {"type": "delta", "text": chunk}
    yield {"type": "done"}


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
