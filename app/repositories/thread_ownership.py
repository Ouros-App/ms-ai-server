from pymongo import ReturnDocument


class ThreadOwnershipStore:
    """Garante atomicamente o usuario dono de cada thread."""

    def __init__(self, database) -> None:
        self.collection = database["thread_owners"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index("thread_id", unique=True)

    async def claim(self, thread_id: str, user_id: str) -> bool:
        document = await self.collection.find_one_and_update(
            {"thread_id": thread_id},
            {"$setOnInsert": {"thread_id": thread_id, "user_id": user_id}},
            projection={"_id": 0, "user_id": 1},
            return_document=ReturnDocument.AFTER,
            upsert=True,
        )
        return document is not None and document.get("user_id") == user_id
