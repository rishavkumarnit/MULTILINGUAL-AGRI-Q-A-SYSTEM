import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .conversations import persist_message
from .database import close_database, connect_database, database_is_connected
from .models import ChatRequest, ChatResponse
from .workflow import LANGUAGE_NAMES, process_question, stream_question

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await connect_database()
    yield
    await close_database()


app = FastAPI(title="Agri Assistant AI Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    normalized = _normalize(request)
    if isinstance(normalized, JSONResponse):
        return normalized
    question, language, conversation_id = normalized

    await persist_message(conversation_id, {
        "role": "user", "content": question, "language": language, "createdAt": datetime.now(timezone.utc),
    })
    result = await process_question(ChatRequest(question=question, language=language, conversationId=conversation_id))
    await persist_message(conversation_id, {
        "role": "assistant",
        "content": result.answer,
        "questionEnglish": result.question_english,
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

    await persist_message(conversation_id, {
        "role": "user", "content": question, "language": language, "createdAt": datetime.now(timezone.utc),
    })

    async def event_source():
        answer_text = ""
        metadata: dict = {}
        normalized_request = ChatRequest(question=question, language=language, conversationId=conversation_id)
        async for event in stream_question(normalized_request):
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
            "crop": metadata.get("crop"),
            "location": metadata.get("location"),
            "source": metadata.get("source"),
            "similarity": metadata.get("similarity"),
            "sources": metadata.get("sources"),
            "createdAt": datetime.now(timezone.utc),
        })

    return StreamingResponse(event_source(), media_type="text/event-stream")
