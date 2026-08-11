import unittest

from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph


class GraphTest(unittest.IsolatedAsyncioTestCase):
    async def test_thread_keeps_messages_without_repeating_agents(self) -> None:
        graph = build_graph(InMemorySaver())
        first = await invoke_graph(
            graph,
            ChatRequest(user_id="user", thread_id="thread", message="teste"),
        )
        second = await invoke_graph(
            graph,
            ChatRequest(user_id="user", thread_id="thread", message="continua"),
        )

        self.assertEqual(first.agents, ["router", "default"])
        self.assertEqual(second.agents, ["router", "default"])

    async def test_thread_rejects_another_user(self) -> None:
        graph = build_graph(InMemorySaver())
        await invoke_graph(graph, ChatRequest(user_id="user-1", thread_id="thread", message="teste"))

        with self.assertRaises(HTTPException) as error:
            await invoke_graph(graph, ChatRequest(user_id="user-2", thread_id="thread", message="teste"))

        self.assertEqual(error.exception.status_code, 403)
