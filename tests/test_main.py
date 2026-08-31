import unittest
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from fastapi import Response
from langgraph.checkpoint.memory import InMemorySaver

from app import main


class MainTest(unittest.IsolatedAsyncioTestCase):
    def request(self) -> SimpleNamespace:
        return SimpleNamespace(
            method="GET",
            url=SimpleNamespace(path="/health"),
            scope={"route": SimpleNamespace(path="/health")},
        )

    async def test_request_middleware_records_success(self) -> None:
        response = Response(status_code=200)

        async def call_next(request):
            return response

        result = await main.log_requests(self.request(), call_next)

        self.assertIs(result, response)

    async def test_request_middleware_records_failure_and_reraises(self) -> None:
        async def call_next(request):
            raise RuntimeError("request failed")

        with self.assertRaises(RuntimeError):
            await main.log_requests(self.request(), call_next)

    async def test_lifespan_isolated_from_external_services(self) -> None:
        client = MagicMock()
        database = MagicMock()
        client.__getitem__.return_value = database
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

        build_graph.assert_called_once_with(
            checkpointer_context.__enter__.return_value,
            memory_store=ANY,
        )
        self.assertIsNotNone(main.app.state.thread_ownership)
        client.close.assert_awaited_once()

    async def test_lifespan_closes_client_when_startup_fails(self) -> None:
        client = MagicMock()
        database = MagicMock()
        client.__getitem__.return_value = database
        client.close = AsyncMock()
        checkpointer_context = MagicMock()
        checkpointer_context.__enter__.return_value = InMemorySaver()
        checkpointer_context.__exit__.return_value = False

        with (
            patch.object(main, "AsyncMongoClient", return_value=client),
            patch.object(main, "get_checkpointer", return_value=checkpointer_context),
            patch.object(main, "build_graph", side_effect=RuntimeError("startup failed")),
            self.assertLogs(main.logger, level="INFO") as logs,
            self.assertRaises(RuntimeError),
        ):
            async with main.lifespan(main.app):
                pass

        client.close.assert_awaited_once()
        self.assertNotIn("database_ready", "\n".join(logs.output))
