"""MongoDB Atlas Vector Search for expert-verified agricultural answers."""

import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from .database import get_database


@dataclass
class SemanticMatch:
    answer_english: str
    similarity: float


async def find_verified_match(question_english: str, crop: str | None, location: str | None) -> SemanticMatch | None:
    """Find a high-confidence, context-compatible reusable answer in Atlas.

    Crop and location are mandatory for direct reuse. The learning project
    accepts both expert-reviewed and explicitly labelled demo records.
    """
    database = get_database()
    if database is None or not crop or not location:
        return None

    client = AsyncOpenAI()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_response = await client.embeddings.create(model=embedding_model, input=question_english)
    query_vector = embedding_response.data[0].embedding

    vector_index = os.getenv("ATLAS_VECTOR_INDEX", "verified_answers_vector")
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 1,
                "filter": {
                    "status": {"$in": ["EXPERT_VERIFIED", "DEMO_VERIFIED"]},
                    "crop": crop.strip().lower(),
                    "location": location.strip().lower(),
                },
            }
        },
        {
            "$project": {
                "answerEnglish": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = await database.verified_answers.aggregate(pipeline)
    async for document in cursor:
        score = float(document["score"])
        threshold = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.88"))
        if score >= threshold:
            return SemanticMatch(answer_english=document["answerEnglish"], similarity=score)
    return None
