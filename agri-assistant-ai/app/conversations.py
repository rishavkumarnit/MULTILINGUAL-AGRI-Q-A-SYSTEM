"""Conversation persistence, moved here from the now-removed Node backend."""

from datetime import datetime, timezone

from .database import get_database

MAX_CONVERSATION_MESSAGES = 50
# How many past turns are actually fed back into the LLM as context per request —
# deliberately much smaller than the storage cap above, since re-sending the whole
# history every turn would multiply token cost with conversation length.
CONTEXT_TURN_LIMIT = 6


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


async def get_conversation_messages(conversation_id: str) -> list[dict]:
    """All persisted messages for this conversation, oldest first. Empty if unknown
    conversation or no database configured."""
    database = get_database()
    if database is None:
        return []
    doc = await database.conversations.find_one({"_id": conversation_id}, {"messages": 1})
    return doc["messages"] if doc else []
