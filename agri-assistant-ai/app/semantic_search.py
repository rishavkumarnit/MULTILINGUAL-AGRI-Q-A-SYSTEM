"""MongoDB Atlas Vector Search for expert-verified agricultural answers."""

import os
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings

from .database import get_database
from .retrievers import AtlasVectorRetriever


@dataclass
class SemanticMatch:
    answer_english: str
    similarity: float


async def find_verified_match(question_english: str, crop: str | None, location: str | None) -> SemanticMatch | None:
    """Find a high-confidence, context-compatible reusable answer in Atlas.

    Crop and location are mandatory for direct reuse. The learning project
    accepts both expert-reviewed and explicitly labelled demo records.
    """
    if get_database() is None or not crop or not location:
        return None

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    retriever = AtlasVectorRetriever(
        collection_name="verified_answers",
        vector_index=os.getenv("ATLAS_VECTOR_INDEX", "verified_answers_vector"),
        embedding=OpenAIEmbeddings(model=embedding_model),
        top_k=1,
        content_field="answerEnglish",
        extra_filter={
            "status": {"$in": ["EXPERT_VERIFIED", "DEMO_VERIFIED"]},
            "crop": crop.strip().lower(),
            "location": location.strip().lower(),
        },
    )
    documents = await retriever.ainvoke(question_english)
    threshold = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.88"))
    for document in documents:
        score = document.metadata["score"]
        if score >= threshold:
            return SemanticMatch(answer_english=document.page_content, similarity=score)
    return None
