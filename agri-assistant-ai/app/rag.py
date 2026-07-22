"""Retrieval-augmented generation over trusted agricultural documents."""

import os
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from .database import get_database
from .retrievers import AtlasVectorRetriever

RAG_PROMPT = PromptTemplate.from_template("Context documents:\n{context}\n\nQuestion: {question}")


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
    if get_database() is None:
        return []

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    retriever = AtlasVectorRetriever(
        collection_name="document_chunks",
        vector_index=os.getenv("DOCUMENT_CHUNKS_VECTOR_INDEX", "document_chunks_vector"),
        embedding=OpenAIEmbeddings(model=embedding_model),
        top_k=top_k or int(os.getenv("RAG_TOP_K", "4")),
        content_field="text",
        metadata_fields=["title"],
    )
    documents = await retriever.ainvoke(question_english)
    threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.75"))
    return [
        RetrievedChunk(text=document.page_content, title=document.metadata["title"], score=document.metadata["score"])
        for document in documents
        if document.metadata["score"] >= threshold
    ]


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
        input=RAG_PROMPT.format(context=context, question=question_english),
    )
    return response.output_text
