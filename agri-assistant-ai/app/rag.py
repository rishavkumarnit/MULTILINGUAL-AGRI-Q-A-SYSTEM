"""Retrieval-augmented generation over trusted agricultural documents."""

import os
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings

from .database import get_database
from .retrievers import AtlasVectorRetriever


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
