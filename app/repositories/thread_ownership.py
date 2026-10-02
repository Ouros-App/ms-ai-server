from datetime import datetime, timezone

from pymongo import ReturnDocument


class ThreadOwnershipStore:
    """Garante atomicamente o usuario dono de cada thread."""

    def __init__(self, database) -> None:
        self.collection = database["thread_owners"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index(
            [("user_id", 1), ("last_activity_at", -1)]
        )

    async def claim(
        self,
        thread_id: str,
        user_id: str,
    ) -> bool:
        now = datetime.now(timezone.utc)
        owner = {
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": now,
        }
        document = await self.collection.find_one_and_update(
            {"thread_id": thread_id},
            {
                "$setOnInsert": owner,
                "$set": {"last_activity_at": now},
            },
            projection={"_id": 0, "user_id": 1},
            return_document=ReturnDocument.AFTER,
            upsert=True,
        )
        return document is not None and document.get("user_id") == user_id

    async def set_title_if_missing(
        self,
        thread_id: str,
        user_id: str,
        title: str,
    ) -> bool:
        normalized_title = " ".join(title.split())[:48]
        if not normalized_title:
            return False
        result = await self.collection.update_one(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "title": {"$exists": False},
            },
            {"$set": {"title": normalized_title}},
        )
        return result.modified_count > 0

    async def is_owned_by_user(self, thread_id: str, user_id: str) -> bool:
        return (
            await self.collection.find_one(
                {"thread_id": thread_id, "user_id": user_id},
                projection={"_id": 1},
            )
            is not None
        )

    async def list_for_user(self, user_id: str, limit: int = 40) -> list[dict]:
        cursor = (
            self.collection.find(
                {"user_id": user_id},
                projection={
                    "_id": 0,
                    "thread_id": 1,
                    "title": 1,
                    "last_activity_at": 1,
                },
            )
            .sort([("last_activity_at", -1), ("thread_id", 1)])
            .limit(limit)
        )
        return [item async for item in cursor]
