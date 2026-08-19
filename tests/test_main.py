import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from app import main


class MainTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_isolated_from_external_services(self) -> None:
        client = MagicMock()
        database = MagicMock()
        collection = MagicMock()
        collection.create_index = AsyncMock()
        client.__getitem__.return_value = database
        database.__getitem__.return_value = collection
        client.close = AsyncMock()

        checkpointer_context = MagicMock()
        checkpointer_context.__enter__.return_value = InMemorySaver()
        checkpointer_context.__exit__.return_value = False

        with (
            patch.object(main, "AsyncMongoClient", return_value=client),
            patch.object(main, "get_checkpointer", return_value=checkpointer_context),
            patch.object(main, "build_graph") as build_graph,
        ):
            async with main.lifespan(main.app):
                self.assertIsNotNone(main.app.state.graph)
                self.assertIs(
                    main.app.state.checkpointer,
                    checkpointer_context.__enter__.return_value,
                )

        collection.create_index.assert_awaited()
        build_graph.assert_called_once_with(
            checkpointer_context.__enter__.return_value,
            memory_store=ANY,
        )
        self.assertIsNotNone(main.app.state.thread_ownership)
        client.close.assert_awaited_once()

    async def test_lifespan_closes_client_when_startup_fails(self) -> None:
        client = MagicMock()
        database = MagicMock()
        collection = MagicMock()
        collection.create_index = AsyncMock(side_effect=RuntimeError("db unavailable"))
        client.__getitem__.return_value = database
        database.__getitem__.return_value = collection
        client.close = AsyncMock()

        with (
            patch.object(main, "AsyncMongoClient", return_value=client),
            self.assertRaises(RuntimeError),
        ):
            async with main.lifespan(main.app):
                pass

        client.close.assert_awaited_once()
