import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .conversations import MAX_CONVERSATION_MESSAGES, get_conversation_messages, persist_message
from .database import close_database, connect_database, database_is_connected
from .models import ChatRequest, ChatResponse
from .workflow import LANGUAGE_NAMES, process_question, stream_question

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await connect_database()
    yield
    await close_database()


# Comma-separated list of allowed origins, e.g. "https://myapp.vercel.app". Defaults to
# "*" so local dev (and initial deploys, before the frontend's final URL is known) keep
# working unchanged; set this explicitly once the production frontend domain is known.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _allowed_origins == "*" else [origin.strip() for origin in _allowed_origins.split(",")]

app = FastAPI(title="Agri Assistant AI Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "database": "connected" if database_is_connected() else "not-configured"}


def _normalize(request: ChatRequest) -> tuple[str, str, str] | JSONResponse:
    """Returns (question, language, conversation_id), or a 400 JSONResponse if invalid.

    Matches the validation the removed Node backend used to do before proxying.
    """
    question = request.question.strip()
    if not question:
        return JSONResponse({"error": "A question is required."}, status_code=400)
    language = request.language if request.language in LANGUAGE_NAMES else "en"
    conversation_id = request.conversation_id or str(uuid.uuid4())
    return question, language, conversation_id


async def _load_conversation(conversation_id: str) -> tuple[list[dict], JSONResponse | None]:
    """Returns (history_messages, None), or ([], error_response) once the conversation
    has hit MAX_CONVERSATION_MESSAGES. A brand-new conversation_id simply has no
    messages yet, so this is a no-op for the first turn."""
    messages = await get_conversation_messages(conversation_id)
    if len(messages) >= MAX_CONVERSATION_MESSAGES:
        return [], JSONResponse(
            {"error": f"This conversation has reached its {MAX_CONVERSATION_MESSAGES}-message limit. Please start a new conversation."},
            status_code=400,
        )
    return messages, None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    normalized = _normalize(request)
    if isinstance(normalized, JSONResponse):
        return normalized
    question, language, conversation_id = normalized

    history, limit_error = await _load_conversation(conversation_id)
    if limit_error:
        return limit_error

    await persist_message(conversation_id, {
        "role": "user", "content": question, "language": language, "createdAt": datetime.now(timezone.utc),
    })
    result = await process_question(
        ChatRequest(question=question, language=language, conversationId=conversation_id), history=history
    )
    await persist_message(conversation_id, {
        "role": "assistant",
        "content": result.answer,
        "questionEnglish": result.question_english,
        "answerEnglish": result.answer_english,
        "crop": result.crop,
        "location": result.location,
        "source": result.source,
        "similarity": result.similarity,
        "sources": result.sources,
        "createdAt": datetime.now(timezone.utc),
    })
    return result.model_copy(update={"conversation_id": conversation_id})


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    normalized = _normalize(request)
    if isinstance(normalized, JSONResponse):
        return normalized
    question, language, conversation_id = normalized

    history, limit_error = await _load_conversation(conversation_id)
    if limit_error:
        return limit_error

    await persist_message(conversation_id, {
        "role": "user", "content": question, "language": language, "createdAt": datetime.now(timezone.utc),
    })

    async def event_source():
        answer_text = ""
        metadata: dict = {}
        normalized_request = ChatRequest(question=question, language=language, conversationId=conversation_id)
        async for event in stream_question(normalized_request, history=history):
            if event["type"] == "metadata":
                metadata = event
                yield f"data: {json.dumps({**event, 'conversationId': conversation_id})}\n\n"
            else:
                if event["type"] == "delta":
                    answer_text += event.get("text", "")
                yield f"data: {json.dumps(event)}\n\n"

        await persist_message(conversation_id, {
            "role": "assistant",
            "content": answer_text,
            "questionEnglish": metadata.get("questionEnglish"),
            "answerEnglish": metadata.get("answerEnglish"),
            "crop": metadata.get("crop"),
            "location": metadata.get("location"),
            "source": metadata.get("source"),
            "similarity": metadata.get("similarity"),
            "sources": metadata.get("sources"),
            "createdAt": datetime.now(timezone.utc),
        })

    return StreamingResponse(event_source(), media_type="text/event-stream")
