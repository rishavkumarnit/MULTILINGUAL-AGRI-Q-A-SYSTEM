"""Conversation persistence, moved here from the now-removed Node backend."""

from datetime import datetime, timezone

from .database import get_database


async def persist_message(conversation_id: str, message: dict) -> None:
    database = get_database()
    if database is None:
        return
    now = datetime.now(timezone.utc)
    await database.conversations.update_one(
        {"_id": conversation_id},
        {
            "$setOnInsert": {"createdAt": now},
            "$set": {"updatedAt": now},
            "$push": {"messages": message},
        },
        upsert=True,
    )
