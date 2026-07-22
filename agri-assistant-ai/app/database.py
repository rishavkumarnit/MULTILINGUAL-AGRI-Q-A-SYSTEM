"""MongoDB Atlas connection shared by FastAPI workflow nodes."""

import os

from dotenv import load_dotenv
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

load_dotenv()

_client: AsyncMongoClient | None = None


async def connect_database() -> None:
    """Connect only when Atlas is configured, keeping local setup usable."""
    global _client
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return
    # _client = AsyncMongoClient(uri, server_api=ServerApi("1", strict=True, deprecation_errors=True))
    _client = AsyncMongoClient(
        uri,
        server_api=ServerApi(
            "1",
            strict=False,
            deprecation_errors=True,
        ),
    )
    await _client.admin.command("ping")


def get_database():
    if not _client:
        return None
    return _client[os.getenv("MONGODB_DATABASE", "agri_assistant")]


def database_is_connected() -> bool:
    return _client is not None


async def close_database() -> None:
    if _client:
        await _client.close()
