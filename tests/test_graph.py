import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import (
    _extract_route,
    _invoke_model,
    build_graph,
    default_agent,
    route_request,
)
from app.agents.model import get_chat_model
from app.agents.prompts import DEFAULT_AGENT_RESPONSE
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph


class GraphTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.previous_groq_key = settings.groq_api_key
        self.previous_nvidia_key = settings.nvidia_api_key
        settings.groq_api_key = None
        settings.nvidia_api_key = None
        get_chat_model.cache_clear()

    def tearDown(self) -> None:
        settings.groq_api_key = self.previous_groq_key
        settings.nvidia_api_key = self.previous_nvidia_key
        get_chat_model.cache_clear()

    async def test_thread_keeps_messages_without_repeating_agents(self) -> None:
        graph = build_graph(InMemorySaver())
        first = await invoke_graph(
            graph,
            ChatRequest(user_id="user", thread_id="thread", message="Como funciona o ranking?"),
            "user",
        )
        second = await invoke_graph(
            graph,
            ChatRequest(user_id="user", thread_id="thread", message="E depois?"),
            "user",
        )

        self.assertEqual(first.agents, ["router", "default"])
        self.assertEqual(second.agents, ["router", "default"])
        self.assertEqual(first.message, DEFAULT_AGENT_RESPONSE)
        self.assertEqual(second.message, DEFAULT_AGENT_RESPONSE)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "thread"}})
        self.assertEqual(
            [message.content for message in snapshot.values["messages"]],
            [
                "Como funciona o ranking?",
                DEFAULT_AGENT_RESPONSE,
                "E depois?",
                DEFAULT_AGENT_RESPONSE,
            ],
        )

    async def test_thread_rejects_another_user(self) -> None:
        graph = build_graph(InMemorySaver())
        await invoke_graph(
            graph,
            ChatRequest(user_id="user-1", thread_id="thread", message="Como funciona o ranking?"),
            "user-1",
        )

        try:
            await invoke_graph(
                graph,
                ChatRequest(user_id="user-2", thread_id="thread", message="Como funciona o ranking?"),
                "user-2",
            )
        except HTTPException as error:
            self.assertEqual(error.status_code, 403)
        else:
            self.fail("A thread deveria rejeitar outro usuario.")


    async def test_default_agent_uses_configured_model(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content='{"route":"ranking"}'),
                AIMessage(content="resposta do modelo"),
            ],
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(InMemorySaver()),
                ChatRequest(
                    user_id="user",
                    thread_id="thread",
                    message="Como funciona o ranking?",
                ),
                "user",
            )

        self.assertEqual(response.message, "resposta do modelo")
        self.assertEqual(response.tools, [])
        self.assertEqual(response.agents, ["router", "ranking"])
        self.assertEqual(model.ainvoke.await_count, 2)

    async def test_router_selects_valid_route_from_model(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"route":"sustainability"}'),
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await route_request(
                {
                    "route": "",
                    "messages": [HumanMessage(content="Como economizar agua?")],
                },
            )

        self.assertEqual(result["route"], "sustainability")
        self.assertEqual(result["agents"], ["router"])

    def test_router_falls_back_for_invalid_model_output(self) -> None:
        self.assertEqual(_extract_route(AIMessage(content="nao e json")), "fallback")
        self.assertEqual(
            _extract_route(AIMessage(content='{"route":"unknown"}')),
            "fallback",
        )
        self.assertEqual(
            _extract_route(AIMessage(content='{"route":"RANKING"}')),
            "ranking",
        )

    async def test_response_reports_tools_used_by_agent(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="memoria consultada"))

        with (
            patch("app.agents.graph.get_chat_model", return_value=model),
            patch(
                "app.agents.graph._invoke_model",
                new=AsyncMock(
                    return_value=(model.ainvoke.return_value, ["recall_user_memories"]),
                ),
            ),
        ):
            response = await invoke_graph(
                build_graph(InMemorySaver()),
                ChatRequest(
                    user_id="user",
                    thread_id="thread-tools",
                    message="Qual foi minha ultima pergunta?",
                ),
                "user",
            )

        self.assertEqual(response.tools, ["recall_user_memories"])

    async def test_model_gets_final_turn_after_tool_limit(self) -> None:
        class MemoryStore:
            async def list(self, user_id: str, limit: int = 20) -> list[str]:
                return []

            async def save(self, user_id: str, memory: str) -> None:
                return None

        tool_call = {"name": "recall_user_memories", "args": {}, "id": "call"}
        tool_enabled_model = Mock()
        tool_enabled_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="", tool_calls=[tool_call]),
            ]
        )
        model = Mock()
        model.bind_tools.return_value = tool_enabled_model
        model.ainvoke = AsyncMock(return_value=AIMessage(content="resposta final"))

        response, tools = await _invoke_model(model, [], MemoryStore(), "user")

        self.assertEqual(response.content, "resposta final")
        self.assertEqual(tools, ["recall_user_memories"])
        model.ainvoke.assert_awaited_once()

    async def test_graph_uses_injected_agent_registry(self) -> None:
        async def specialist(state):
            return {
                "agents": [*state["agents"], "specialist"],
                "messages": [AIMessage(content="resposta especializada")],
            }

        graph = build_graph(
            InMemorySaver(),
            agents={"default": default_agent, "specialist": specialist},
        )
        result = await graph.ainvoke(
            {"messages": [], "user_id": "user", "route": "specialist", "agents": []},
            config={"configurable": {"thread_id": "specialist-thread"}},
        )

        self.assertEqual(result["messages"][-1].content, "resposta especializada")
        self.assertEqual(result["agents"], ["router", "specialist"])
