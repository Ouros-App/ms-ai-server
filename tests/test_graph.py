import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.agents.prompts import DEFAULT_AGENT_RESPONSE
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
        snapshot = await graph.aget_state({"configurable": {"thread_id": "thread"}})
        self.assertEqual(
            [message.content for message in snapshot.values["messages"]],
            ["teste", DEFAULT_AGENT_RESPONSE, "continua", DEFAULT_AGENT_RESPONSE],
        )

    async def test_thread_rejects_another_user(self) -> None:
        graph = build_graph(InMemorySaver())
        await invoke_graph(graph, ChatRequest(user_id="user-1", thread_id="thread", message="teste"))

        try:
            await invoke_graph(graph, ChatRequest(user_id="user-2", thread_id="thread", message="teste"))
        except HTTPException as error:
            self.assertEqual(error.status_code, 403)
        else:
            self.fail("A thread deveria rejeitar outro usuario.")


    async def test_default_agent_uses_configured_model(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="resposta do modelo"))

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(InMemorySaver()),
                ChatRequest(user_id="user", thread_id="thread", message="teste"),
            )

        self.assertEqual(response.message, "resposta do modelo")
        model.ainvoke.assert_awaited_once()
