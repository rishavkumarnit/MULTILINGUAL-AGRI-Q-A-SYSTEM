"""Insert a reusable learning-project answer and its embedding into Atlas."""

import argparse
import asyncio
import os
from datetime import datetime, timezone

from langchain_openai import OpenAIEmbeddings

from app.database import close_database, connect_database, get_database


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a reusable agricultural answer.")
    parser.add_argument("--question", required=True, help="Approved English question")
    parser.add_argument("--answer", required=True, help="Approved English answer")
    parser.add_argument("--crop", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--status", choices=("DEMO_VERIFIED", "EXPERT_VERIFIED"), default="DEMO_VERIFIED")
    arguments = parser.parse_args()

    await connect_database()
    database = get_database()
    if database is None:
        raise RuntimeError("MONGODB_URI must be configured in .env")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    embedding_vector = await embeddings.aembed_query(arguments.question)
    document = {
        "questionEnglish": arguments.question,
        "answerEnglish": arguments.answer,
        "crop": arguments.crop.strip().lower(),
        "location": arguments.location.strip().lower(),
        "status": arguments.status,
        "embedding": embedding_vector,
        "embeddingModel": embedding_model,
        "updatedAt": datetime.now(timezone.utc),
    }
    await database.verified_answers.update_one(
        {"questionEnglish": document["questionEnglish"], "crop": document["crop"], "location": document["location"]},
        {"$set": document, "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}},
        upsert=True,
    )
    await close_database()
    print("Verified answer ingested.")


if __name__ == "__main__":
    asyncio.run(main())
