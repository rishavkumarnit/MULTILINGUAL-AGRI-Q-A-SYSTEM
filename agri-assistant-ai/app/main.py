from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import close_database, connect_database, database_is_connected
from .models import ChatRequest, ChatResponse
from .workflow import process_question

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
