import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import (
    _RESET_TOOLS,
    _deterministic_routes,
    _extract_route,
    _invoke_model,
    _merge_tools,
    build_graph,
    default_agent,
    route_request,
)
from app.agents.model import get_chat_model
from app.agents.prompts import DEFAULT_AGENT_RESPONSE, FALLBACK_RESPONSE
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

    def test_tool_reducer_distinguishes_reset_from_no_tools(self) -> None:
        self.assertEqual(_merge_tools(["get_user_context"], []), ["get_user_context"])
        self.assertEqual(_merge_tools(["old"], [_RESET_TOOLS]), [])
        self.assertEqual(_merge_tools(["old"], [_RESET_TOOLS, "new"]), ["new"])

    async def test_invoke_model_rejects_reset_sentinel_as_tool_name(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        reset_tool = Mock()
        reset_tool.name = _RESET_TOOLS

        response, used_tools = await _invoke_model(
            model,
            [],
            None,
            "user",
            [reset_tool],
        )

        self.assertEqual(response.content, "ok")
        self.assertEqual(used_tools, [])
        model.bind_tools.assert_not_called()

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

        self.assertEqual(first.agents, ["router", "ranking", "default"])
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
        """Usa o modelo configurado na síntese final do agente default."""
        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content='{"status":"ok","facts":["nivel prata"],"recommendations":[],"missing_data":[],"sources":["ranking_mcp"]}',
                ),
                AIMessage(content="resposta sintetizada"),
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

        self.assertEqual(response.message, "resposta sintetizada")
        self.assertEqual(response.tools, [])
        self.assertEqual(response.agents, ["router", "ranking", "default"])
        self.assertEqual(model.ainvoke.await_count, 2)
        model.bind_tools.assert_not_called()

    async def test_router_selects_valid_route_from_model(self) -> None:
        """Mantém o roteamento por modelo quando não há intenção explícita."""
        model = Mock()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"route":"sustainability"}'),
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await route_request(
                {
                    "route": "",
                    "messages": [
                        HumanMessage(content="Qual abordagem devo priorizar neste caso?")
                    ],
                },
            )

        self.assertEqual(result["route"], "sustainability")
        self.assertEqual(result["agents"], ["router"])

    def test_deterministic_router_matches_clear_project_intents(self) -> None:
        """Identifica intenções claras sem depender de uma chamada ao roteador."""
        self.assertEqual(
            _deterministic_routes(
                HumanMessage(
                    content="Quais indicadores de sustentabilidade devo acompanhar?"
                )
            ),
            ["sustainability"],
        )
        self.assertEqual(
            _deterministic_routes(HumanMessage(content="Falhou a sincronizacao offline")),
            ["support"],
        )
        self.assertEqual(
            _deterministic_routes(
                HumanMessage(content="Quero ver meu ranking e reduzir o consumo de agua")
            ),
            ["ranking", "sustainability"],
        )

    async def test_fallback_route_returns_safe_clarifying_response(self) -> None:
        """Retorna apenas uma pergunta segura para mensagens ambíguas."""
        result = await default_agent(
            {
                "routes": ["fallback"],
                "agents": ["router"],
                "tools": [],
                "specialist_results": [{"agent": "fallback", "status": "ok"}],
                "input_guardrail": {"allowed": True},
            }
        )

        self.assertEqual(result["messages"][0].content, FALLBACK_RESPONSE)

    async def test_fallback_route_skips_specialist_in_graph(self) -> None:
        """Encaminha fallback diretamente ao default sem chamada extra."""
        model = Mock()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(content='{"route":"fallback"}')
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(InMemorySaver()),
                ChatRequest(
                    user_id="user",
                    thread_id="fallback-thread",
                    message="ajuda",
                ),
                "user",
            )

        self.assertEqual(response.message, FALLBACK_RESPONSE)
        self.assertEqual(response.agents, ["router", "default"])
        self.assertEqual(model.ainvoke.await_count, 1)

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
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content='{"route":"ranking"}'),
                AIMessage(content="resposta sintetizada"),
            ],
        )

        with (
            patch("app.agents.graph.get_chat_model", return_value=model),
            patch(
                "app.agents.graph._invoke_model",
                new=AsyncMock(
                    return_value=(
                        AIMessage(
                            content='{"status":"ok","facts":["memoria consultada"],"recommendations":[],"missing_data":[],"sources":["memory"]}',
                        ),
                        ["recall_user_memories"],
                    ),
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
        self.assertEqual(response.agents, ["router", "ranking", "default"])

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

    async def test_tool_model_error_falls_back_without_tool_schema(self) -> None:
        mcp_tool = Mock(name="get_user_context")
        mcp_tool.name = "get_user_context"
        tool_enabled_model = Mock()
        tool_enabled_model.ainvoke = AsyncMock(side_effect=RuntimeError("tool format"))
        model = Mock()
        model.bind_tools.return_value = tool_enabled_model
        model.ainvoke = AsyncMock(return_value=AIMessage(content='{"status":"ok"}'))

        response, tools = await _invoke_model(model, [], None, "42", [mcp_tool])

        self.assertEqual(response.content, '{"status":"ok"}')
        self.assertEqual(tools, [])
        model.ainvoke.assert_awaited_once()

    async def test_invoke_model_executes_external_mcp_tool(self) -> None:
        mcp_tool = Mock(name="get_user_context")
        mcp_tool.name = "get_user_context"
        mcp_tool.ainvoke = AsyncMock(return_value='{"user_id":"42"}')
        tool_call = {"name": mcp_tool.name, "args": {"user_id": "42"}, "id": "mcp-call"}

        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content='{"status":"ok"}'),
            ],
        )
        model = Mock()
        model.bind_tools.return_value = bound_model

        response, tools = await _invoke_model(model, [], None, "42", [mcp_tool])

        self.assertEqual(response.content, '{"status":"ok"}')
        self.assertEqual(tools, ["get_user_context"])
        mcp_tool.ainvoke.assert_awaited_once_with({"user_id": "42"})

    async def test_graph_uses_injected_agent_registry(self) -> None:
        async def specialist(state):
            return {
                "agents": [*state["agents"], "specialist"],
                "specialist_results": [
                    {
                        "agent": "specialist",
                        "status": "ok",
                        "facts": ["resultado estruturado"],
                        "recommendations": [],
                        "missing_data": [],
                        "sources": [],
                    },
                ],
            }

        graph = build_graph(
            InMemorySaver(),
            agents={"default": default_agent, "specialist": specialist},
        )
        result = await graph.ainvoke(
            {"messages": [], "user_id": "user", "route": "specialist", "agents": []},
            config={"configurable": {"thread_id": "specialist-thread"}},
        )

        self.assertEqual(
            result["messages"][-1].content,
            DEFAULT_AGENT_RESPONSE,
        )
        self.assertEqual(result["agents"], ["router", "specialist", "default"])

    async def test_specialist_result_is_kept_out_of_conversation_messages(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content='{"status":"ok","facts":["funciona offline"],"recommendations":[],"missing_data":[],"sources":[]}',
                ),
                AIMessage(content="O aplicativo funciona offline."),
            ],
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            graph = build_graph(InMemorySaver())
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Como funciona offline?")],
                    "user_id": "user",
                    "route": "faq",
                    "agents": [],
                    "tools": [],
                    "input_guardrail": {"allowed": True, "category": "APROVADO", "message": ""},
                    "specialist_results": [],
                },
                config={"configurable": {"thread_id": "json-thread"}},
            )

        self.assertEqual([message.content for message in result["messages"]], [
            "Como funciona offline?",
            "O aplicativo funciona offline.",
        ])
        self.assertEqual(result["specialist_results"][0]["facts"], ["funciona offline"])

    async def test_router_can_fan_out_to_multiple_specialists_and_synthesize_once(self) -> None:
        async def route_plan(state):
            return {
                "route": "faq",
                "routes": ["faq", "ranking"],
                "agents": ["router"],
                "tools": [],
                "specialist_results": [],
            }

        async def specialist(name, state):
            return {
                "agents": [*state["agents"], name],
                "specialist_results": [
                    {
                        "agent": name,
                        "status": "ok",
                        "facts": [name],
                        "recommendations": [],
                        "missing_data": [],
                        "sources": [],
                    },
                ],
            }

        async def faq(state):
            return await specialist("faq", state)

        async def ranking(state):
            return await specialist("ranking", state)

        async def synth(state):
            return {
                "agents": [*state["agents"], "default"],
                "messages": [AIMessage(content="sintese multiagente")],
            }

        with patch(
            "app.agents.graph.route_request",
            new=route_plan,
        ):
            graph = build_graph(
                InMemorySaver(),
                agents={
                    "default": synth,
                    "faq": faq,
                    "ranking": ranking,
                },
            )
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Como estou no ranking e como uso o app?")],
                    "user_id": "user",
                    "route": "",
                    "routes": [],
                    "agents": [],
                    "tools": [],
                    "specialist_results": [],
                },
                config={"configurable": {"thread_id": "fanout-thread"}},
            )

        self.assertEqual(result["messages"][-1].content, "sintese multiagente")
        self.assertEqual(result["agents"][0], "router")
        self.assertEqual(result["agents"][-1], "default")
        self.assertEqual(set(result["agents"][1:-1]), {"faq", "ranking"})
        self.assertEqual(
            {item["agent"] for item in result["specialist_results"]},
            {"faq", "ranking"},
        )

    async def test_mcp_tools_are_injected_only_into_specialists(self) -> None:
        """Mantém tools MCP restritas ao especialista escolhido."""
        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="sintese final"),
            ],
        )
        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='{"status":"ok","facts":["dado MCP"],"recommendations":[],"missing_data":[],"sources":["ranking_mcp"]}',
            ),
        )
        model.bind_tools.return_value = bound_model
        mcp_tool = Mock()

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(
                    InMemorySaver(),
                    specialist_tools={"ranking": [mcp_tool]},
                ),
                ChatRequest(
                    user_id="user",
                    thread_id="mcp-thread",
                    message="Como estou no ranking?",
                ),
                "user",
            )

        self.assertEqual(response.message, "sintese final")
        model.bind_tools.assert_called_once_with([mcp_tool])
