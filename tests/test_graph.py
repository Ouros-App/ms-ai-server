import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import (
    _RESET_TOOLS,
    _collect_pending_state,
    _conversation_is_personal_request,
    _deterministic_routes,
    _extract_period_days,
    _extract_route,
    _invoke_model,
    _is_pending_followup,
    _merge_tools,
    _resolve_local_routes,
    _route_update,
    _tool_args_trace,
    _tool_result_content,
    _tool_result_trace,
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
            _extract_period_days("30", ["periodo de analise"]),
            30,
        )
        self.assertIsNone(_extract_period_days("30"))
        self.assertIsNone(_extract_period_days("13 meses"))

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
            "Qual e o meu selo?",
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
        self.assertIsNone(
            _deterministic_routes(
                HumanMessage(content="Quero ver meu ranking e reduzir o consumo de agua")
            )
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
