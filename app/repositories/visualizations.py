from datetime import datetime, timezone


class ChatVisualizationStore:
    """Persist chart HTML separately from LangGraph checkpoints."""

    def __init__(self, database) -> None:
        self.collection = database["chat_visualizations"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index(
            [
                ("thread_id", 1),
                ("user_id", 1),
                ("visualization_id", 1),
            ],
            unique=True,
        )

    async def save(
        self,
        thread_id: str,
        user_id: str,
        visualization_id: str,
        visualizations: list[dict],
    ) -> None:
        await self.collection.update_one(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "visualization_id": visualization_id,
            },
            {
                "$set": {
                    "visualizations": visualizations,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def get_many(
        self,
        thread_id: str,
        user_id: str,
        visualization_ids: list[str],
    ) -> dict[str, list[dict]]:
        if not visualization_ids:
            return {}
        cursor = self.collection.find(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "visualization_id": {"$in": visualization_ids},
            },
            projection={"_id": 0, "visualization_id": 1, "visualizations": 1},
        )
        return {
            item["visualization_id"]: item["visualizations"]
            async for item in cursor
        }
