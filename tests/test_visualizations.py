import unittest
from unittest.mock import AsyncMock, Mock

from app.repositories.visualizations import ChatVisualizationStore


class ChatVisualizationStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_save_scopes_visuals_to_owner_and_message(self) -> None:
        collection = Mock()
        collection.update_one = AsyncMock()
        store = ChatVisualizationStore({"chat_visualizations": collection})
        visualizations = [{"type": "ouros_dashboard", "charts": []}]

        await store.save("thread-1", "user-1", "viz-1", visualizations)

        args = collection.update_one.await_args
        self.assertEqual(
            args.args[0],
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "visualization_id": "viz-1",
            },
        )
        self.assertEqual(args.args[1]["$set"]["visualizations"], visualizations)
        self.assertTrue(args.kwargs["upsert"])

    async def test_get_many_reads_only_requested_thread_owner_and_messages(self) -> None:
        class Cursor:
            def __aiter__(self):
                async def values():
                    yield {
                        "visualization_id": "viz-1",
                        "visualizations": [{"type": "ouros_dashboard"}],
                    }

                return values()

        collection = Mock()
        collection.find = Mock(return_value=Cursor())
        store = ChatVisualizationStore({"chat_visualizations": collection})

        result = await store.get_many("thread-1", "user-1", ["viz-1"])

        self.assertEqual(result, {"viz-1": [{"type": "ouros_dashboard"}]})
        collection.find.assert_called_once_with(
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "visualization_id": {"$in": ["viz-1"]},
            },
            projection={"_id": 0, "visualization_id": 1, "visualizations": 1},
        )
