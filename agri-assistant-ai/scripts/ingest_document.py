"""Chunk, embed, and upsert a trusted agricultural document into Atlas."""

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from app.chunking import chunk_text
from app.database import close_database, connect_database, get_database


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a trusted agricultural document for RAG.")
    parser.add_argument("--path", required=True, help="Path to the document (markdown or text)")
    parser.add_argument("--title", required=True, help="Human-readable document title, used as a citation")
    parser.add_argument("--crop", default=None, help="Optional primary crop metadata")
    parser.add_argument("--location", default=None, help="Optional location metadata")
    arguments = parser.parse_args()

    text = Path(arguments.path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks produced; is the document empty?")

    await connect_database()
    database = get_database()
    if database is None:
        raise RuntimeError("MONGODB_URI must be configured in .env")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = AsyncOpenAI()
    embedding_response = await client.embeddings.create(model=embedding_model, input=chunks)

    now = datetime.now(timezone.utc)
    documents = [
        {
            "title": arguments.title,
            "source": Path(arguments.path).name,
            "crop": arguments.crop.strip().lower() if arguments.crop else None,
            "location": arguments.location.strip().lower() if arguments.location else None,
            "chunkIndex": index,
            "text": chunk,
            "embedding": embedding_response.data[index].embedding,
            "embeddingModel": embedding_model,
            "createdAt": now,
        }
        for index, chunk in enumerate(chunks)
    ]

    await database.document_chunks.delete_many({"title": arguments.title})
    await database.document_chunks.insert_many(documents)
    await close_database()
    print(f"Ingested {len(documents)} chunks for '{arguments.title}'.")


if __name__ == "__main__":
    asyncio.run(main())
