import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .database import close_database, connect_database, database_is_connected
from .models import ChatRequest, ChatResponse
from .workflow import process_question, stream_question

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await connect_database()
    yield
    await close_database()


app = FastAPI(title="Agri Assistant AI Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "database": "connected" if database_is_connected() else "not-configured"}


@app.post("/internal/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await process_question(request)


@app.post("/internal/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_source():
        async for event in stream_question(request):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
