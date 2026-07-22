"""Retrieval-augmented generation over trusted agricultural documents."""

import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from .database import get_database


@dataclass
class RetrievedChunk:
    text: str
    title: str
    score: float


async def find_relevant_chunks(question_english: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Semantic search over ingested document chunks, unfiltered by crop/location.

    Unlike verified-answer reuse, documents are general-purpose knowledge, so
    retrieval relies on similarity alone rather than exact context matching.
    """
    database = get_database()
    if database is None:
        return []

    client = AsyncOpenAI()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_response = await client.embeddings.create(model=embedding_model, input=question_english)
    query_vector = embedding_response.data[0].embedding

    limit = top_k or int(os.getenv("RAG_TOP_K", "4"))
    vector_index = os.getenv("DOCUMENT_CHUNKS_VECTOR_INDEX", "document_chunks_vector")
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": limit,
            }
        },
        {
            "$project": {
                "text": 1,
                "title": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.75"))
    cursor = await database.document_chunks.aggregate(pipeline)
    chunks = []
    async for document in cursor:
        score = float(document["score"])
        if score >= threshold:
            chunks.append(RetrievedChunk(text=document["text"], title=document["title"], score=score))
    return chunks


async def generate_rag_answer(question_english: str, chunks: list[RetrievedChunk]) -> str:
    """Generate an answer grounded only in the retrieved document chunks."""
    context = "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in chunks)
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            "You are an agricultural assistant. Answer the farmer's question using only the "
            "provided context documents. Do not invent facts not present in the context. "
            "If the context does not fully answer the question, say what is missing rather than "
            "guessing. Return only the answer in plain English."
        ),
        input=f"Context documents:\n{context}\n\nQuestion: {question_english}",
    )
    return response.output_text
