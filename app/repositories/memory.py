import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class UserMemoryStore:
    """Persistencia de memorias duraveis compartilhadas entre threads do usuario."""

    def __init__(self, database) -> None:
        self.collection = database["user_memories"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index(
            [("user_id", 1), ("memory", 1)],
            unique=True,
        )

    async def list(self, user_id: str, limit: int = 20) -> list[str]:
        cursor = (
            self.collection.find(
                {"user_id": user_id},
                {"_id": 0, "memory": 1},
            )
            .sort("updated_at", -1)
            .limit(limit)
        )
        documents = await cursor.to_list(length=limit)
        logger.info("memory_recalled count=%d", len(documents))
        return [document["memory"] for document in documents]

    async def save(self, user_id: str, memory: str) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"user_id": user_id, "memory": memory},
            {
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "user_id": user_id,
                    "memory": memory,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        logger.info("memory_saved")
