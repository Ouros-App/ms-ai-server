import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import (
    _DASHBOARD_IMPLICIT_REQUEST_PATTERN,
    _RESET_TOOLS,
    _collect_pending_state,
    _conversation_is_personal_request,
    _dashboard_confirmation,
    _dashboard_period_days,
    _dashboard_period_days_for_state,
    _deterministic_routes,
    _execute_specialist,
    _execute_tool_call,
    _extract_period_days,
    _extract_route,
    _invoke_model,
    _is_dashboard_creation_request,
    _is_pending_followup,
    _is_visualization_subject_reply,
    _merge_tools,
    _normalize_specialist_result,
    _prefetch_dashboard_catalog,
    _resolve_jev_route,
    _resolve_local_routes,
    _route_update,
    _run_agent,
    _tool_args_trace,
    _tool_result_content,
    _tool_result_trace,
    build_graph,
    default_agent,
    route_request,
)
from app.agents.mcp import capture_mcp_visualizations
from app.agents.model import get_chat_model
from app.agents.prompts import (
    DEFAULT_AGENT_RESPONSE,
    FALLBACK_RESPONSE,
    GREETING_RESPONSE,
    IDENTITY_RESPONSE,
)
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

    def test_chart_followup_keeps_period_only_for_visualization_continuation(self) -> None:
        """Keep earlier periods only for an active chart continuation."""
        history = [
            HumanMessage(content="Como foi meu consumo nos últimos 3 meses?"),
            AIMessage(content="O período consultado foi de 90 dias."),
            HumanMessage(content="Pode gerar um gráfico?"),
        ]

        self.assertEqual(
            _dashboard_period_days_for_state(
                {
                    "messages": history,
                    "route_source": "context",
                    "routes": ["visualization"],
                },
                "Pode gerar um gráfico?",
            ),
            90,
        )
        self.assertEqual(
            _dashboard_period_days_for_state(
                {
                    "messages": history,
                    "route_source": "deterministic",
                    "routes": ["visualization"],
                },
                "3 dias",
            ),
            3,
        )
        self.assertEqual(
            _dashboard_period_days_for_state(
                {
                    "messages": [HumanMessage(content="Gere um gráfico")],
                    "route_source": "context",
                    "routes": ["visualization"],
                },
                "Gere um gráfico",
            ),
            30,
        )

    def test_visualization_subject_replies_are_limited_to_chart_prompts(self) -> None:
        """Reject unrelated questions when the chart topic is still missing."""
        missing = ["assunto do gráfico"]

        self.assertFalse(_is_dashboard_creation_request(None))
        self.assertFalse(_is_visualization_subject_reply(None, missing))
        self.assertFalse(_is_visualization_subject_reply("água", []))
        self.assertFalse(_is_visualization_subject_reply("Qual consumo?", missing))
        self.assertFalse(_is_visualization_subject_reply("cancela", missing))
        self.assertTrue(_is_visualization_subject_reply("consumo de água", missing))
        self.assertTrue(_is_pending_followup("consumo de água", missing))
        self.assertFalse(_is_pending_followup("Como faço isso?", missing))
        self.assertFalse(
            _is_pending_followup(
                "Qual abordagem devo priorizar neste caso?",
                missing,
            )
        )
        self.assertTrue(_is_pending_followup("E depois?", ["periodo de analise"]))

    async def test_dashboard_catalog_prefetch_handles_missing_failed_and_invalid_tools(self) -> None:
        missing_message, missing_tools = await _prefetch_dashboard_catalog([])
        self.assertEqual(missing_tools, [])
        self.assertIn("não está disponível", missing_message["content"])

        failing_tool = SimpleNamespace(
            name="get_custom_dashboard_catalog",
            ainvoke=AsyncMock(side_effect=RuntimeError("offline")),
        )
        failed_message, failed_tools = await _prefetch_dashboard_catalog([failing_tool])
        self.assertEqual(failed_tools, [])
        self.assertIn("consulta do catálogo de gráficos falhou", failed_message["content"])

        for invalid_result in (None, [], {"charts": []}, {"charts": "invalid"}):
            tool = SimpleNamespace(
                name="get_custom_dashboard_catalog",
                ainvoke=AsyncMock(return_value=invalid_result),
            )
            invalid_message, used_tools = await _prefetch_dashboard_catalog([tool])
            self.assertEqual(used_tools, [])
            self.assertIn("dados inválidos", invalid_message["content"])

    async def test_dashboard_catalog_prefetch_returns_authorized_catalog_context(self) -> None:
        catalog = {"charts": [{"chart_id": "daily-water", "title": "Consumo diário"}]}
        tool = SimpleNamespace(
            name="get_custom_dashboard_catalog",
            ainvoke=AsyncMock(return_value=catalog),
        )

        message, used_tools = await _prefetch_dashboard_catalog([tool])

        self.assertEqual(used_tools, ["get_custom_dashboard_catalog"])
        self.assertIn('"chart_id": "daily-water"', message["content"])
        self.assertIn("periodo e a granularidade sao distintos", message["content"])
        self.assertIn("frequencia dos registros", message["content"])
        self.assertIn("janela que atravessa meses nao pede agrupamento mensal", message["content"])
        self.assertIn("render_as` muda apenas a forma visual", message["content"])
        tool.ainvoke.assert_awaited_once_with({})

    async def test_dashboard_prefetch_timeout_disables_catalog(self) -> None:
        tool = SimpleNamespace(
            name="get_custom_dashboard_catalog",
            ainvoke=AsyncMock(side_effect=TimeoutError()),
        )

        with patch("app.agents.graph.settings.mcp_tool_timeout_seconds", 0):
            message, used_tools = await _prefetch_dashboard_catalog([tool])

        self.assertEqual(used_tools, [])
        self.assertIn("consulta do catálogo de gráficos falhou", message["content"])

    async def test_visualization_prefetches_catalog_and_keeps_creation_tool(self) -> None:
        catalog = SimpleNamespace(
            name="get_custom_dashboard_catalog",
            ainvoke=AsyncMock(return_value={"charts": [{"chart_id": "daily-water"}]}),
        )
        create = SimpleNamespace(name="create_custom_dashboard")
        observed = {}

        async def invoke(_model, messages, _store, _user_id, tools):
            observed["messages"] = messages
            observed["tools"] = tools
            return AIMessage(content='{"status":"ok","facts":[]}'), []

        state = {
            "messages": [HumanMessage(content="gere um gráfico de consumo")],
            "user_id": "user",
            "routes": ["visualization"],
            "decision_source": "jev",
            "decision_tools": ["create_custom_dashboard"],
        }
        with (
            patch("app.agents.graph._load_agent_mcp_tools", new=AsyncMock(return_value=[catalog, create])),
            patch("app.agents.graph._prefetch_consumption_summary", new=AsyncMock(return_value=(None, [], None))),
            patch("app.agents.graph._invoke_model", side_effect=invoke),
        ):
            result, used_tools, personal_request = await _execute_specialist(
                state, "visualization prompt", "visualization", "gere um gráfico de consumo",
                Mock(), [], None,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(used_tools, ["get_custom_dashboard_catalog"])
        self.assertFalse(personal_request)
        self.assertEqual([tool.name for tool in observed["tools"]], ["create_custom_dashboard"])
        self.assertIn(
            "daily-water",
            " ".join(
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
                for message in observed["messages"]
            ),
        )

    async def test_visualization_removes_creation_tool_when_catalog_is_unavailable(self) -> None:
        create = SimpleNamespace(name="create_custom_dashboard")
        observed = {}

        async def invoke(_model, _messages, _store, _user_id, tools):
            observed["tools"] = tools
            return AIMessage(content='{"status":"ok","facts":[]}'), []

        state = {
            "messages": [HumanMessage(content="gere um gráfico de consumo")],
            "user_id": "user",
            "routes": ["visualization"],
            "decision_source": "jev",
            "decision_tools": ["create_custom_dashboard"],
        }
        with (
            patch("app.agents.graph._load_agent_mcp_tools", new=AsyncMock(return_value=[create])),
            patch("app.agents.graph._prefetch_consumption_summary", new=AsyncMock(return_value=(None, [], None))),
            patch("app.agents.graph._invoke_model", side_effect=invoke),
        ):
            result, used_tools, _ = await _execute_specialist(
                state, "visualization prompt", "visualization", "gere um gráfico de consumo",
                Mock(), [], None,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(used_tools, [])
        self.assertEqual(observed["tools"], [])

    def test_explicit_chart_request_routes_to_visualization_with_context(self) -> None:
        """Route chart requests to visualization and inherit their domain."""
        missing = ["assunto do gráfico"]
        request = HumanMessage(content="Pode gerar um gráfico?")
        routes, source = _resolve_local_routes(
            {
                "messages": [request],
                "last_routes": ["sustainability"],
            }
        )
        self.assertEqual((routes, source), (["visualization"], "context"))

        routes, source = _resolve_local_routes(
            {"messages": [HumanMessage(content="Gere um gráfico de água")]}
        )
        self.assertEqual(
            (routes, source),
            (["sustainability", "visualization"], "deterministic"),
        )

        routes, source = _resolve_local_routes(
            {
                "messages": [
                    HumanMessage(content="Como foi meu desempenho no último mês?")
                ]
            }
        )
        self.assertEqual((routes, source), (["ranking", "visualization"], "deterministic"))

        routes, source = _resolve_local_routes(
            {
                "messages": [HumanMessage(content="água")],
                "pending_by_route": {"visualization": missing},
            }
        )
        self.assertEqual((routes, source), (["visualization"], "pending"))

    async def test_jev_cannot_override_explicit_chart_route_without_chart_tool(self) -> None:
        """Keep explicit chart creation out of a Jev route without the tool."""
        decision = SimpleNamespace(
            agent="default",
            tools=[],
            confidence=0.95,
            needs_mcp=False,
            needs_analytics=False,
        )
        outcome = SimpleNamespace(accepted=True, decision=decision)
        state = {
            "messages": [HumanMessage(content="Pode gerar um gráfico?")],
            "last_routes": ["sustainability"],
        }

        with (
            patch.object(settings, "jev_enabled", True),
            patch.object(settings, "jev_shadow_mode", False),
            patch("app.agents.graph._DECISION_SERVICE.decide", new=AsyncMock(return_value=outcome)),
        ):
            routes, source, metadata = await _resolve_jev_route(
                state,
                ["visualization"],
                "context",
            )

        self.assertEqual((routes, source), (["visualization"], "context"))
        self.assertEqual(metadata["decision_source"], "")

    async def test_jev_chart_route_preserves_followup_period_context(self) -> None:
        """Preserve local chart follow-up sources after Jev confirms visualization."""
        decision = SimpleNamespace(
            agent="visualization",
            tools=["create_custom_dashboard"],
            confidence=0.95,
            needs_mcp=True,
            needs_analytics=False,
        )
        outcome = SimpleNamespace(accepted=True, decision=decision)

        with (
            patch.object(settings, "jev_enabled", True),
            patch.object(settings, "jev_shadow_mode", False),
            patch("app.agents.graph._DECISION_SERVICE.decide", new=AsyncMock(return_value=outcome)),
        ):
            routes, source, metadata = await _resolve_jev_route(
                {
                    "messages": [HumanMessage(content="Pode gerar um gráfico?")],
                    "last_routes": ["sustainability"],
                },
                ["visualization"],
                "context",
            )

        self.assertEqual((routes, source), (["visualization"], "context"))
        self.assertEqual(metadata["decision_source"], "jev")

        with (
            patch.object(settings, "jev_enabled", True),
            patch.object(settings, "jev_shadow_mode", False),
            patch("app.agents.graph._DECISION_SERVICE.decide", new=AsyncMock(return_value=outcome)),
        ):
            routes, source, metadata = await _resolve_jev_route(
                {
                    "messages": [HumanMessage(content="consumo de água")],
                    "last_routes": ["visualization"],
                },
                ["visualization"],
                "pending",
            )

        self.assertEqual((routes, source), (["visualization"], "pending"))
        self.assertEqual(metadata["decision_source"], "jev")

    async def test_implicit_performance_request_adds_visualization_route(self) -> None:
        """Append charts to implicit requests about performance over time."""
        with patch(
            "app.agents.graph._resolve_jev_route",
            new=AsyncMock(return_value=(["ranking"], "context", {})),
        ):
            result = await route_request(
                {
                    "messages": [
                        HumanMessage(
                            content="Como foi meu desempenho no último mês?"
                        )
                    ],
                    "input_guardrail": {"allowed": True},
                    "route": "",
                    "pending_by_route": {},
                }
            )

        self.assertEqual(result["routes"], ["ranking", "visualization"])

    async def test_unknown_tool_call_gets_a_matching_error_tool_message(self) -> None:
        conversation = []
        used_tools: list[str] = []

        await _execute_tool_call(
            {"name": "not_allowed", "id": "call-1", "args": {}},
            {},
            conversation,
            used_tools,
        )

        self.assertEqual(used_tools, [])
        self.assertEqual(len(conversation), 1)
        self.assertIsInstance(conversation[0], ToolMessage)
        self.assertEqual(conversation[0].tool_call_id, "call-1")
        self.assertIn("Ferramenta nao disponivel", conversation[0].content)

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
        self.assertEqual(second.agents, ["router", "ranking", "default"])
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

    async def test_greeting_and_identity_skip_router_model_and_specialists(self) -> None:
        graph = build_graph(InMemorySaver())

        greeting = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="quick-greeting",
                message="bom dia",
            ),
            "user",
        )
        identity = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="quick-identity",
                message="quem eh vc?",
            ),
            "user",
        )

        self.assertEqual(greeting.agents, ["router", "default"])
        self.assertEqual(identity.agents, ["router", "default"])
        self.assertIn("Midas", greeting.message)
        self.assertIn("Midas", identity.message)

    async def test_default_agent_handles_quick_routes_after_jev_selection(self) -> None:
        """Keep greeting and identity replies when Jev selects the default route."""
        cases = (
            ("Bom dia", GREETING_RESPONSE),
            ("Quem é você?", IDENTITY_RESPONSE),
        )
        for message, expected in cases:
            result = await default_agent(
                {
                    "messages": [HumanMessage(content=message)],
                    "route_source": "jev",
                    "routes": ["default"],
                    "specialist_results": [],
                    "agents": ["router"],
                    "tools": [],
                }
            )

            with self.subTest(message=message):
                self.assertEqual(result["messages"][0].content, expected)

    async def test_default_agent_answers_general_message_without_specialists(self) -> None:
        """Use the general model when Jev selects default without specialist data."""
        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="Posso ajudar com isso."),
                AIMessage(content="STATUS: APROVADO\nRESPOSTA:\nPosso ajudar com isso."),
            ],
        )
        state = {
            "messages": [HumanMessage(content="Pode me ajudar?")],
            "route_source": "jev",
            "routes": ["default"],
            "specialist_results": [],
            "agents": ["router"],
            "tools": [],
        }

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await default_agent(state)

        self.assertEqual(result["messages"][0].content, "Posso ajudar com isso.")
        self.assertEqual(model.ainvoke.await_count, 2)
        synthesis_messages = model.ainvoke.await_args_list[0].args[0]
        self.assertIn(
            "Nenhum especialista forneceu dados",
            synthesis_messages[-1]["content"],
        )

    async def test_default_agent_contains_general_model_failures(self) -> None:
        """Keep a model exception from aborting a default-route chat request."""
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        state = {
            "messages": [HumanMessage(content="Pode me ajudar?")],
            "route_source": "jev",
            "routes": ["default"],
            "specialist_results": [],
            "agents": ["router"],
            "tools": [],
        }

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await default_agent(state)

        self.assertEqual(result["messages"][0].content, DEFAULT_AGENT_RESPONSE)

    async def test_default_agent_confirms_created_dashboard_without_model_call(self) -> None:
        state = {
            "messages": [HumanMessage(content="Gere um gráfico")],
            "route_source": "jev",
            "routes": ["visualization"],
            "specialist_results": [
                {"agent": "visualization", "status": "ok", "_dashboard_created": True}
            ],
            "agents": ["router", "visualization"],
            "tools": ["create_custom_dashboard"],
        }

        with (
            patch("app.agents.graph.get_chat_model", side_effect=AssertionError),
            capture_mcp_visualizations() as visualizations,
        ):
            visualizations.append(
                {
                    "type": "ouros_dashboard",
                    "charts": [{"title": "Consumo mensal de água"}],
                }
            )
            result = await default_agent(state)

        self.assertEqual(
            result["messages"][0].content,
            "Pronto, gerei o gráfico solicitado: Consumo mensal de água. "
            "O título indica agrupamento mensal.",
        )
        self.assertTrue(result["messages"][0].additional_kwargs["visualization_id"])

    def test_dashboard_confirmation_qualifies_monthly_grain_from_title(self) -> None:
        for title in (
            "Consumo mensal de água",
            "Consumos mensais de água",
            "Consumo por mês",
            "Monthly water consumption",
        ):
            with self.subTest(title=title):
                confirmation = _dashboard_confirmation(
                    [{"charts": [{"title": title}]}]
                )
                self.assertIn("O título indica agrupamento mensal.", confirmation)

    async def test_specialist_failure_is_collected_without_aborting_the_graph(self) -> None:
        """Turn a specialist model exception into a result for safe synthesis."""
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        state = {
            "messages": [HumanMessage(content="Como funciona o aplicativo?")],
            "user_id": "user",
            "agents": ["router"],
            "input_guardrail": {"allowed": True},
        }

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await _run_agent(state, "prompt", "faq")

        self.assertEqual(result["specialist_results"][0]["status"], "error")

    async def test_contextual_followup_inherits_route_and_explicit_topic_wins(self) -> None:
        """Carry context only for referential follow-ups, not explicit topic changes."""
        graph = build_graph(InMemorySaver())

        first = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="context-thread",
                message="Como funciona o ranking?",
            ),
            "user",
        )
        followup = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="context-thread",
                message="E depois?",
            ),
            "user",
        )
        switched = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="context-thread",
                message="Agora quero analisar meu consumo de agua",
            ),
            "user",
        )

        self.assertEqual(first.agents, ["router", "ranking", "default"])
        self.assertEqual(followup.agents, ["router", "ranking", "default"])
        self.assertEqual(
            switched.agents,
            ["router", "sustainability", "default"],
        )

    async def test_structured_pending_answer_keeps_specialist_route(self) -> None:
        """A compact answer to requested data must continue the active task."""

        async def sustainability(state):
            latest = state["messages"][-1].content
            needs_period = "30 dias" not in latest.lower()
            return {
                "agents": [*state["agents"], "sustainability"],
                "specialist_results": [
                    {
                        "agent": "sustainability",
                        "status": "needs_input" if needs_period else "ok",
                        "facts": [] if needs_period else ["periodo recebido"],
                        "recommendations": [],
                        "missing_data": ["periodo de analise"] if needs_period else [],
                        "sources": [],
                    }
                ],
            }

        async def synth(state):
            needs_input = any(
                result.get("status") == "needs_input"
                for result in state.get("specialist_results", [])
            )
            return {
                "agents": [*state["agents"], "default"],
                "messages": [
                    AIMessage(
                        content="Qual periodo?" if needs_input else "Periodo aplicado."
                    )
                ],
            }

        graph = build_graph(
            InMemorySaver(),
            agents={"default": synth, "sustainability": sustainability},
        )
        first = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="pending-thread",
                message="Quero analisar meu consumo de agua",
            ),
            "user",
        )
        first_snapshot = await graph.aget_state(
            {"configurable": {"thread_id": "pending-thread"}}
        )
        second = await invoke_graph(
            graph,
            ChatRequest(
                user_id="user",
                thread_id="pending-thread",
                message="30 dias na minha fazenda",
            ),
            "user",
        )
        second_snapshot = await graph.aget_state(
            {"configurable": {"thread_id": "pending-thread"}}
        )

        self.assertEqual(first.agents, ["router", "sustainability", "default"])
        self.assertEqual(
            first_snapshot.values["pending_routes"],
            ["sustainability"],
        )
        self.assertEqual(second.agents, ["router", "sustainability", "default"])
        self.assertEqual(second.message, "Periodo aplicado.")
        self.assertEqual(second_snapshot.values["pending_routes"], [])
        self.assertEqual(second_snapshot.values["pending_missing_data"], [])

    def test_pending_reply_must_match_the_requested_slot(self) -> None:
        self.assertTrue(
            _is_pending_followup(
                HumanMessage(content="30 dias"),
                ["periodo de analise"],
            )
        )
        self.assertTrue(
            _is_pending_followup(
                HumanMessage(content="1 ciclo"),
                ["ciclo ou periodo de analise"],
            )
        )
        self.assertIsNone(_extract_period_days("1 ciclo", ["periodo de analise"]))
        self.assertFalse(
            _is_pending_followup(
                HumanMessage(content="Quero cadastrar 2 propriedades"),
                ["periodo de analise"],
            )
        )

    def test_quick_turns_preserve_pending_state_but_cancel_clears_it(self) -> None:
        greeting_update = _route_update(["default"], "greeting")
        cancelled_update = _route_update(["fallback"], "cancelled")

        self.assertNotIn("pending_by_route", greeting_update)
        self.assertEqual(cancelled_update["pending_routes"], [])
        self.assertEqual(cancelled_update["pending_missing_data"], [])
        self.assertEqual(cancelled_update["pending_by_route"], {})
        self.assertEqual(cancelled_update["pending_personal_routes"], [])

    def test_same_route_new_task_retires_pending_personal_context(self) -> None:
        state = {
            "messages": [
                HumanMessage(
                    content="Como funciona o consumo de agua no aplicativo?"
                )
            ],
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "pending_personal_routes": ["sustainability"],
        }

        update = _route_update(
            ["sustainability"],
            "deterministic",
            state,
        )

        self.assertEqual(update["pending_routes"], [])
        self.assertEqual(update["pending_missing_data"], [])
        self.assertEqual(update["pending_by_route"], {})
        self.assertEqual(update["pending_personal_routes"], [])

    def test_same_route_slot_answer_keeps_pending_personal_context(self) -> None:
        state = {
            "messages": [HumanMessage(content="consumo nos ultimos 30 dias")],
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "pending_personal_routes": ["sustainability"],
        }

        update = _route_update(
            ["sustainability"],
            "deterministic",
            state,
        )

        self.assertNotIn("pending_routes", update)
        self.assertNotIn("pending_personal_routes", update)

    def test_explicit_topic_switch_retires_unrelated_pending_task(self) -> None:
        state = {
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "pending_personal_routes": ["sustainability"],
        }

        switched = _route_update(
            ["ranking"],
            "deterministic",
            state,
        )
        greeting = _route_update(
            ["default"],
            "greeting",
            state,
        )

        self.assertEqual(switched["pending_routes"], [])
        self.assertEqual(switched["pending_missing_data"], [])
        self.assertEqual(switched["pending_by_route"], {})
        self.assertEqual(switched["pending_personal_routes"], [])
        self.assertNotIn("pending_by_route", greeting)

    def test_cancel_does_not_revive_previous_route(self) -> None:
        state = {
            "messages": [HumanMessage(content="Esquece isso")],
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "last_routes": ["sustainability"],
        }

        self.assertEqual(
            _resolve_local_routes(state),
            (["fallback"], "cancelled"),
        )


    def test_cancel_command_can_name_pending_route(self) -> None:
        state = {
            "messages": [HumanMessage(content="Esquece o ranking")],
            "pending_routes": ["ranking"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"ranking": ["periodo de analise"]},
            "last_routes": ["ranking"],
        }

        self.assertEqual(
            _resolve_local_routes(state),
            (["fallback"], "cancelled"),
        )

    def test_product_cancel_language_is_not_treated_as_task_cancellation(self) -> None:
        state = {
            "messages": [HumanMessage(content="Como cancelar uma notificacao?")],
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "last_routes": ["sustainability"],
        }

        routes, source = _resolve_local_routes(state)

        self.assertNotEqual(source, "cancelled")
        self.assertNotEqual(routes, ["fallback"])

    def test_pending_data_remains_isolated_by_route(self) -> None:
        routes, missing, by_route, personal_routes = _collect_pending_state(
            [
                {
                    "agent": "sustainability",
                    "status": "needs_input",
                    "missing_data": ["periodo de analise"],
                    "_personal_data_required": True,
                },
                {
                    "agent": "ranking",
                    "status": "needs_input",
                    "missing_data": ["estado do ranking"],
                    "_personal_data_required": False,
                },
            ]
        )

        self.assertEqual(routes, ["sustainability", "ranking"])
        self.assertEqual(
            by_route,
            {
                "sustainability": ["periodo de analise"],
                "ranking": ["estado do ranking"],
            },
        )
        self.assertEqual(missing, ["periodo de analise", "estado do ranking"])
        self.assertEqual(personal_routes, ["sustainability"])

    def test_personal_requirement_survives_a_bare_period_followup(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="Como esta o consumo da minha fazenda?"),
                AIMessage(content="Qual periodo?"),
                HumanMessage(content="bom dia"),
                AIMessage(content="Oi!"),
                HumanMessage(content="30"),
            ],
            "pending_routes": ["sustainability"],
            "pending_missing_data": ["periodo de analise"],
            "pending_by_route": {"sustainability": ["periodo de analise"]},
            "pending_personal_routes": ["sustainability"],
        }

        self.assertTrue(
            _conversation_is_personal_request(
                state,
                "sustainability",
                "30",
            )
        )

    def test_period_parser_handles_natural_followups_without_guessing(self) -> None:
        self.assertEqual(_extract_period_days("30 dias na minha fazenda"), 30)
        self.assertEqual(_extract_period_days("ultima semana"), 7)
        self.assertEqual(_extract_period_days("ultimo mes"), 30)
        self.assertEqual(
            _extract_period_days("ultimos 30 dias ate hoje"),
            30,
        )
        self.assertEqual(
            _extract_period_days("30", ["periodo de analise"]),
            30,
        )
        self.assertIsNone(_extract_period_days("30"))
        self.assertIsNone(_extract_period_days("13 meses"))

    def test_dashboard_period_understands_feminine_relative_periods(self) -> None:
        for message in ("desempenho da ultima semana", "desempenho da semana passada"):
            with self.subTest(message=message):
                self.assertEqual(_dashboard_period_days(message), 7)
                self.assertRegex(
                    message,
                    _DASHBOARD_IMPLICIT_REQUEST_PATTERN,
                )

    def test_legacy_league_names_are_not_hardcoded_into_routing(self) -> None:
        self.assertIsNone(
            _deterministic_routes(HumanMessage(content="Estou no cobre?"))
        )

    def test_lot_mentions_only_route_to_faq_when_the_intent_is_app_usage(self) -> None:
        self.assertEqual(
            _deterministic_routes(
                HumanMessage(content="Quanto gasta de agua um lote com 5 mil frangos?")
            ),
            ["sustainability"],
        )
        self.assertEqual(
            _deterministic_routes(HumanMessage(content="Como cadastrar um lote?")),
            ["faq"],
        )


    def test_identity_and_auth_messages_route_without_losing_context(self) -> None:
        self.assertEqual(
            _deterministic_routes(HumanMessage(content="quem eh vc?")),
            ["faq"],
        )
        self.assertEqual(
            _deterministic_routes(HumanMessage(content="minha senha nao funciona")),
            ["support"],
        )
        self.assertEqual(
            _deterministic_routes(HumanMessage(content="Como acesso meu ranking?")),
            ["ranking"],
        )

    def test_personal_ranking_indicators_are_detected_as_personal_data(self) -> None:
        """Detect personal ranking, score, level and badge queries without choosing storage."""
        from app.agents.graph import _is_personal_data_request

        for message in (
            "Qual e o meu ranking?",
            "Qual e a minha pontuacao?",
            "Qual e o meu nivel?",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    _is_personal_data_request("ranking", message)
                )

    async def test_personal_consumption_query_prefetches_domain_summary(self) -> None:
        """Fetch only the scoped aggregate needed for an explicit personal period."""
        summary_tool = Mock()
        summary_tool.name = "get_consumption_summary"
        summary_tool.ainvoke = AsyncMock(
            return_value={
                "user_type": "farm_owner",
                "user_id": 42,
                "authorized": True,
                "period_days": 30,
                "water_unit": "hydrometer_reading_delta",
                "energy_unit": "kWh",
                "summaries": [
                    {
                        "id_farm": 11,
                        "water_meter_delta": 80,
                        "energy_consumption_kwh": 120,
                    }
                ],
            }
        )
        provider = Mock()
        provider.tools_for = AsyncMock(return_value=[summary_tool])

        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content=(
                        '{"status":"ok","facts":["consumo encontrado"],'
                        '"recommendations":[],"missing_data":[],"sources":["mcp"]}'
                    )
                ),
                AIMessage(content="Seu consumo foi encontrado."),
            ]
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(InMemorySaver(), mcp_provider=provider),
                ChatRequest(
                    user_id="42",
                    thread_id="personal-data-thread",
                    message=(
                        "Como foi o consumo de agua da minha fazenda nos ultimos 30 dias?"
                    ),
                ),
                "42",
                principal_token="signed-user-token",
            )

        summary_tool.ainvoke.assert_awaited_once_with({"period_days": 30})
        self.assertIn("get_consumption_summary", response.tools)
        specialist_messages = model.ainvoke.await_args_list[0].args[0]
        system_context = "\n".join(
            str(message.get("content", ""))
            for message in specialist_messages
            if isinstance(message, dict)
        )
        self.assertIn("Dados pessoais autenticados", system_context)
        self.assertIn("water_meter_delta", system_context)

    async def test_personal_query_does_not_request_manual_data_without_farm_scope(
        self,
    ) -> None:
        """Treat missing authenticated farm scope as backend state, not user input."""
        summary_tool = Mock()
        summary_tool.name = "get_consumption_summary"
        summary_tool.ainvoke = AsyncMock(
            return_value={
                "user_type": "farm_owner",
                "user_id": 42,
                "authorized": False,
                "reason": "no_farm_scope",
                "period_days": 30,
                "water_unit": "hydrometer_reading_delta",
                "energy_unit": "kWh",
                "summaries": [],
            }
        )
        provider = Mock()
        provider.tools_for = AsyncMock(return_value=[summary_tool])

        model = Mock()
        model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content=(
                        '{"status":"needs_input","facts":[],'
                        '"recommendations":[],"missing_data":'
                        '["dados de consumo de agua"],"sources":[]}'
                    )
                ),
                AIMessage(content="Os dados da sua fazenda estao indisponiveis."),
                AIMessage(
                    content=(
                        "STATUS: APROVADO\nRESPOSTA:\n"
                        "Os dados da sua fazenda estao indisponiveis."
                    )
                ),
            ]
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            response = await invoke_graph(
                build_graph(InMemorySaver(), mcp_provider=provider),
                ChatRequest(
                    user_id="42",
                    thread_id="no-farm-scope-thread",
                    message="Como esta o consumo de agua da minha fazenda nos ultimos 30 dias?",
                ),
                "42",
                principal_token="signed-user-token",
            )

        self.assertEqual(
            response.message,
            "Os dados da sua fazenda estao indisponiveis.",
        )
        specialist_context = model.ainvoke.await_args_list[1].args[0][-1]["content"]
        self.assertNotIn("dados de consumo de agua", specialist_context)
        self.assertIn('"status":"error"', specialist_context)
        self.assertIn('"missing_data":[]', specialist_context)

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
                AIMessage(
                    content=(
                        "STATUS: APROVADO\nRESPOSTA:\n"
                        "resposta sintetizada"
                    )
                ),
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
        self.assertEqual(model.ainvoke.await_count, 3)
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
        self.assertIsNone(
            _deterministic_routes(
                HumanMessage(content="Quero ver meu ranking e reduzir o consumo de agua")
            )
        )

    def test_legacy_product_feature_is_not_hardcoded_into_routing(self) -> None:
        message = HumanMessage(content="Onde vejo meus selos antigos?")
        state = {
            "messages": [message],
            "pending_routes": [],
            "pending_missing_data": [],
            "pending_by_route": {},
            "pending_personal_routes": [],
        }

        self.assertIsNone(_deterministic_routes(message))
        self.assertFalse(
            _conversation_is_personal_request(
                state,
                "faq",
                "Onde vejo meus selos antigos?",
            )
        )
        self.assertIsNone(
            _deterministic_routes(HumanMessage(content="Qual e o preco do ouro?"))
        )

    def test_overlapping_intents_use_semantic_router_unless_explicitly_compound(self) -> None:
        self.assertIsNone(
            _deterministic_routes(
                HumanMessage(content="Meu ranking deu erro")
            )
        )
        self.assertIsNone(
            _deterministic_routes(
                HumanMessage(content="Onde vejo meu consumo no app?")
            )
        )
        self.assertIsNone(
            _deterministic_routes(
                HumanMessage(content="Quero ver meu ranking e reduzir meu consumo")
            )
        )

    async def test_semantic_router_can_choose_multiple_agents_for_real_multi_intent(self) -> None:
        model = Mock()
        model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='{"routes":["ranking","sustainability"]}'
            ),
        )

        with patch("app.agents.graph.get_chat_model", return_value=model):
            result = await route_request(
                {
                    "route": "",
                    "messages": [
                        HumanMessage(
                            content="Quero ver meu ranking e reduzir meu consumo de agua"
                        )
                    ],
                }
            )

        self.assertEqual(result["routes"], ["ranking", "sustainability"])
        self.assertEqual(result["route_source"], "model")
        model.ainvoke.assert_awaited_once()

    async def test_deterministic_route_works_without_router_model(self) -> None:
        """Mantém a rota clara mesmo sem modelo disponível para o roteador."""
        with patch("app.agents.graph.get_chat_model", return_value=None):
            result = await route_request(
                {
                    "route": "",
                    "messages": [HumanMessage(content="Como funciona o ranking?")],
                }
            )

        self.assertEqual(result["routes"], ["ranking"])

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
                    message="Sou produtor e preciso de orientacao.",
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

    def test_tool_argument_trace_keeps_shape_not_values(self) -> None:
        trace = _tool_args_trace(
            {
                "query": "conteudo sensivel da fazenda",
                "period_days": 30,
            }
        )

        self.assertEqual(trace["arg_count"], 2)
        self.assertEqual(trace["keys"], ["period_days", "query"])
        self.assertNotIn("conteudo sensivel", str(trace))
        self.assertNotIn("30", str(trace))

    def test_tool_results_are_bounded_and_trace_safe(self) -> None:
        payload = {"secret_business_value": "x" * 13_000}

        trace = _tool_result_trace(payload)
        model_content = _tool_result_content(payload)

        self.assertEqual(trace["type"], "dict")
        self.assertIn("secret_business_value", trace["keys"])
        self.assertNotIn("x" * 100, str(trace))
        self.assertIn('"status":"truncated"', model_content)
        self.assertLess(len(model_content), 12_500)

    async def test_tool_timeout_is_returned_to_model_without_hanging_request(self) -> None:
        async def slow_tool(_args):
            await asyncio.sleep(0.05)
            return {"status": "ok"}

        mcp_tool = Mock(name="get_user_context")
        mcp_tool.name = "get_user_context"
        mcp_tool.ainvoke = AsyncMock(side_effect=slow_tool)
        tool_call = {"name": mcp_tool.name, "args": {}, "id": "mcp-timeout"}

        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(
                    content=(
                        '{"status":"error","facts":[],"recommendations":[],'
                        '"missing_data":[],"sources":[]}'
                    )
                ),
            ],
        )
        model = Mock()
        model.bind_tools.return_value = bound_model

        with patch.object(settings, "mcp_tool_timeout_seconds", 0.01):
            response, tools = await _invoke_model(
                model,
                [],
                None,
                "42",
                [mcp_tool],
            )

        self.assertEqual(tools, ["get_user_context"])
        self.assertIn('"status":"error"', response.content)
        second_messages = bound_model.ainvoke.await_args_list[1].args[0]
        tool_message = next(
            message for message in second_messages if isinstance(message, ToolMessage)
        )
        self.assertIn("temporariamente indisponivel", tool_message.content)

    async def test_tool_failure_is_returned_to_model_without_crashing_graph(self) -> None:
        mcp_tool = Mock(name="get_user_context")
        mcp_tool.name = "get_user_context"
        mcp_tool.ainvoke = AsyncMock(side_effect=RuntimeError("database secret detail"))
        tool_call = {"name": mcp_tool.name, "args": {}, "id": "mcp-call"}

        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content='{"status":"error","facts":[],"recommendations":[],"missing_data":[],"sources":[]}'),
            ],
        )
        model = Mock()
        model.bind_tools.return_value = bound_model

        response, tools = await _invoke_model(model, [], None, "42", [mcp_tool])

        self.assertEqual(tools, ["get_user_context"])
        self.assertIn('"status":"error"', response.content)
        second_messages = bound_model.ainvoke.await_args_list[1].args[0]
        tool_message = next(
            message for message in second_messages if isinstance(message, ToolMessage)
        )
        self.assertIn("temporariamente indisponivel", tool_message.content)
        self.assertNotIn("database secret detail", tool_message.content)

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

    async def test_dashboard_creation_skips_model_follow_up(self) -> None:
        dashboard_tool = Mock(name="create_custom_dashboard")
        dashboard_tool.name = "create_custom_dashboard"
        dashboard_tool.ainvoke = AsyncMock(return_value={"title": "Consumo"})
        tool_call = {
            "name": dashboard_tool.name,
            "args": {"title": "Consumo", "charts": []},
            "id": "dashboard-call",
        }
        tool_enabled_model = Mock()
        tool_enabled_model.ainvoke = AsyncMock(
            return_value=AIMessage(content="", tool_calls=[tool_call])
        )
        model = Mock()
        model.bind_tools.return_value = tool_enabled_model

        response, tools = await _invoke_model(
            model, [], None, "42", [dashboard_tool]
        )

        result = _normalize_specialist_result(response)
        self.assertTrue(result["_dashboard_created"])
        self.assertEqual(tools, ["create_custom_dashboard"])
        tool_enabled_model.ainvoke.assert_awaited_once()

    def test_model_cannot_claim_dashboard_was_created(self) -> None:
        result = _normalize_specialist_result(
            AIMessage(
                content=(
                    '{"status":"ok","facts":[],"recommendations":[],'
                    '"missing_data":[],"sources":[],"_dashboard_created":true}'
                )
            )
        )

        self.assertNotIn("_dashboard_created", result)

    async def test_failed_dashboard_creation_keeps_normal_model_flow(self) -> None:
        dashboard_tool = Mock(name="create_custom_dashboard")
        dashboard_tool.name = "create_custom_dashboard"
        dashboard_tool.ainvoke = AsyncMock(side_effect=RuntimeError("remote failure"))
        tool_call = {
            "name": dashboard_tool.name,
            "args": {"title": "Consumo", "charts": []},
            "id": "dashboard-call",
        }
        tool_enabled_model = Mock()
        tool_enabled_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content='{"status":"error"}'),
            ]
        )
        model = Mock()
        model.bind_tools.return_value = tool_enabled_model

        response, tools = await _invoke_model(
            model, [], None, "42", [dashboard_tool]
        )

        self.assertEqual(response.content, '{"status":"error"}')
        self.assertNotIn("_dashboard_created", response.content)
        self.assertEqual(tools, ["create_custom_dashboard"])
        self.assertEqual(tool_enabled_model.ainvoke.await_count, 2)

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
                AIMessage(
                    content=(
                        "STATUS: APROVADO\nRESPOSTA:\n"
                        "O aplicativo funciona offline."
                    )
                ),
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
                AIMessage(
                    content="STATUS: APROVADO\nRESPOSTA:\nsintese final"
                ),
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
