import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from langchain_core.messages import HumanMessage

from app.agents import graph
from app.agents.graph import _model_mcp_tools
from app.core.config import settings
from app.decision.fallback import DeterministicFallbackProvider
from app.decision.jev_provider import (
    DecisionProviderError,
    DecisionTimeoutError,
    InvalidDecisionError,
    JevDecisionProvider,
)
from app.decision.models import (
    DecisionInput,
    DecisionOutcome,
    MidasDecision,
    ProviderResult,
)
from app.decision.service import DecisionService


def _decision_input() -> DecisionInput:
    """Build the bounded fixture state used by provider and service tests."""
    return DecisionInput(
        message="Mostre o consumo de água no último mês",
        available_agents=["faq", "sustainability", "default"],
        agent_tools={
            "faq": ["search_knowledge"],
            "sustainability": ["get_consumption_summary"],
            "default": [],
        },
        context={"has_analytics": False},
    )


def _provider_body(*, agent: str = "sustainability") -> dict:
    """Build a valid System One response fixture for the selected route."""
    return {
        "model": "jev-latest",
        "answers": {
            "agent": {"type": "choice", "choice": agent, "confidence": 0.94},
            "needs_analytics": {"type": "noul", "noul": 0.1},
            "tool_0": {"type": "noul", "noul": 0.9},
            "tool_1": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 30, "output_tokens": 5},
    }


class JevProviderTests(unittest.TestCase):
    def test_provider_parses_route_tools_confidence_and_usage(self) -> None:
        """Parse a valid response into route, tools, confidence, and usage."""
        result = JevDecisionProvider._parse_response(_provider_body(), _decision_input())

        self.assertEqual(result.decision.agent, "sustainability")
        self.assertEqual(result.decision.tools, ["get_consumption_summary"])
        self.assertTrue(result.decision.needs_mcp)
        self.assertFalse(result.decision.needs_analytics)
        self.assertEqual(result.decision.confidence, 0.94)
        self.assertEqual((result.input_tokens, result.output_tokens), (30, 5))

    def test_provider_rejects_unavailable_route_or_missing_answer(self) -> None:
        """Reject a route outside the candidate set and omitted decisions."""
        invalid_route_body = _provider_body(agent="other")
        state = _decision_input()
        with self.assertRaises(InvalidDecisionError) as invalid_route:
            JevDecisionProvider._parse_response(invalid_route_body, state)
        self.assertEqual(invalid_route.exception.reason, "agent_unavailable")

        body = _provider_body()
        del body["answers"]["tool_0"]
        state_with_missing_answer = _decision_input()
        with self.assertRaises(InvalidDecisionError) as missing_answer:
            JevDecisionProvider._parse_response(body, state_with_missing_answer)
        self.assertEqual(missing_answer.exception.reason, "probability_missing")

    def test_provider_classifies_malformed_decision_fields(self) -> None:
        """Assign a specific reason to every malformed provider response field."""
        cases = (
            ([], "response_not_object"),
            ({"answers": []}, "answers_missing"),
            ({"answers": {}}, "agent_answer_missing"),
        )
        for body, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                with self.assertRaises(InvalidDecisionError) as raised:
                    JevDecisionProvider._parse_response(body, _decision_input())
                self.assertEqual(raised.exception.reason, expected_reason)

        invalid_fields = (
            ("agent", "confidence", 1.1, "confidence_invalid"),
            ("tool_0", "noul", 1.1, "probability_invalid"),
        )
        for answer_name, field_name, value, expected_reason in invalid_fields:
            body = _provider_body()
            body["answers"][answer_name][field_name] = value
            with self.subTest(reason=expected_reason):
                with self.assertRaises(InvalidDecisionError) as raised:
                    JevDecisionProvider._parse_response(body, _decision_input())
                self.assertEqual(raised.exception.reason, expected_reason)

        invalid_bodies = []
        body_without_usage = _provider_body()
        del body_without_usage["usage"]
        invalid_bodies.append((body_without_usage, "usage_missing"))
        body_with_invalid_usage = _provider_body()
        body_with_invalid_usage["usage"]["input_tokens"] = -1
        invalid_bodies.append((body_with_invalid_usage, "usage_invalid"))
        body_without_model = _provider_body()
        del body_without_model["model"]
        invalid_bodies.append((body_without_model, "model_missing"))
        body_with_invalid_schema = _provider_body()
        body_with_invalid_schema["model"] = "m" * 101
        invalid_bodies.append((body_with_invalid_schema, "decision_schema_invalid"))
        for body, expected_reason in invalid_bodies:
            with self.subTest(reason=expected_reason):
                with self.assertRaises(InvalidDecisionError) as raised:
                    JevDecisionProvider._parse_response(body, _decision_input())
                self.assertEqual(raised.exception.reason, expected_reason)

    def test_analytics_question_and_decision_when_backend_is_available(self) -> None:
        """Ask Jev about analytics and honor its answer when supported."""
        state = _decision_input().model_copy(update={"context": {"has_analytics": True}})
        body = _provider_body()
        body["answers"]["needs_analytics"] = {"type": "noul", "noul": 0.9}

        questions = JevDecisionProvider._questions(state)
        result = JevDecisionProvider._parse_response(body, state)

        self.assertIn("needs_analytics", questions)
        self.assertTrue(result.decision.needs_analytics)

    def test_provider_state_contains_no_user_or_history_fields(self) -> None:
        """Keep identity, credentials, and conversation history out of state."""
        provider_state = _decision_input().to_provider_state()

        self.assertEqual(
            set(provider_state),
            {"message", "available_agents", "available_tools", "agent_tools", "context"},
        )
        self.assertNotIn("user_id", provider_state)
        self.assertNotIn("messages", provider_state)
        self.assertNotIn("jwt", provider_state)

    def test_graph_decision_input_anonymizes_message_and_omits_identity(self) -> None:
        """Anonymize the current message and omit internal identity fields."""
        decision_input = graph._build_decision_input(
            {
                "user_id": "internal-user-123",
                "messages": [HumanMessage(content="histórico que não deve ser enviado")],
                "last_routes": ["sustainability"],
            },
            "Meu consumo, contato pessoa@example.com",
        )

        self.assertIsNotNone(decision_input)
        self.assertNotIn("pessoa@example.com", decision_input.message)
        self.assertNotIn("internal-user-123", str(decision_input.to_provider_state()))
        self.assertNotIn("messages", decision_input.to_provider_state())


class JevProviderRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_api_key_has_a_specific_provider_error(self) -> None:
        """Explain an absent provider credential without attempting a request."""
        provider = JevDecisionProvider(
            api_key=None,
            base_url="https://typesafe.example",
            model="jev-test",
            timeout_seconds=1,
        )

        with self.assertRaises(DecisionProviderError) as raised:
            await provider.decide(_decision_input())

        self.assertEqual(raised.exception.reason, "api_key_missing")

    async def test_request_uses_system_one_contract_and_minimized_state(self) -> None:
        """Send a single minimized request to the configured System One path."""
        captured: dict[str, object] = {}

        async def respond(request: httpx.Request) -> httpx.Response:
            """Capture the request and return a valid fixture response."""
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_provider_body())

        provider = JevDecisionProvider(
            api_key="secret-test-key",
            base_url="https://typesafe.example",
            model="jev-test",
            timeout_seconds=1,
            transport=httpx.MockTransport(respond),
        )

        await provider.decide(_decision_input())

        payload = captured["payload"]
        self.assertEqual(captured["url"], "https://typesafe.example/v1/systemone")
        self.assertEqual(captured["authorization"], "Bearer secret-test-key")
        self.assertEqual(payload["model"], "jev-test")
        self.assertNotIn("user_id", payload["state"])
        self.assertNotIn("messages", payload["state"])
        self.assertEqual(
            set(payload["questions"]["agent"]["criteria"]),
            {"faq", "sustainability", "default"},
        )
        self.assertNotIn("needs_mcp", payload["questions"])
        self.assertNotIn("needs_analytics", payload["questions"])
        self.assertIn(
            "sustainability",
            payload["questions"]["tool_0"]["instructions"],
        )

    async def test_tool_selection_is_limited_to_the_selected_route(self) -> None:
        """Ignore a tool vote that belongs to a different selected agent."""
        body = _provider_body(agent="default")
        body["answers"]["tool_0"]["noul"] = 0.95

        result = JevDecisionProvider._parse_response(body, _decision_input())

        self.assertEqual(result.decision.agent, "default")
        self.assertEqual(result.decision.tools, [])
        self.assertFalse(result.decision.needs_mcp)

    async def test_unavailable_analytics_is_not_requested_or_selected(self) -> None:
        """Do not ask Jev for an analytics strategy without that backend."""
        body = _provider_body()
        body["answers"]["needs_analytics"]["noul"] = 0.99

        result = JevDecisionProvider._parse_response(body, _decision_input())
        questions = JevDecisionProvider._questions(_decision_input())

        self.assertFalse(result.decision.needs_analytics)
        self.assertNotIn("needs_analytics", questions)

    async def test_timeout_and_provider_http_errors_are_classified(self) -> None:
        """Map timeouts and retryable HTTP statuses to provider errors."""
        state = _decision_input()

        async def timeout(_request: httpx.Request) -> httpx.Response:
            """Simulate an HTTP transport timeout."""
            raise httpx.ReadTimeout("provider timeout")

        timeout_provider = JevDecisionProvider(
            "key", "https://typesafe.example", "jev-test", 0.1,
            transport=httpx.MockTransport(timeout),
        )
        with self.assertRaises(DecisionTimeoutError):
            await timeout_provider.decide(state)

        for status_code in (429, 500):
            async def fail(_request: httpx.Request, code=status_code) -> httpx.Response:
                """Return one failing status code from the API."""
                return httpx.Response(code)

            provider = JevDecisionProvider(
                "key", "https://typesafe.example", "jev-test", 0.1,
                transport=httpx.MockTransport(fail),
            )
            with (
                self.subTest(status_code=status_code),
                self.assertRaises(DecisionProviderError),
            ):
                await provider.decide(state)

        async def connection_error(request: httpx.Request) -> httpx.Response:
            """Simulate a failed network connection to the provider."""
            raise httpx.ConnectError("connection refused", request=request)

        transport_provider = JevDecisionProvider(
            "key",
            "https://typesafe.example",
            "jev-test",
            0.1,
            transport=httpx.MockTransport(connection_error),
        )
        with self.assertRaises(DecisionProviderError):
            await transport_provider.decide(state)

    async def test_invalid_json_and_incomplete_responses_are_rejected(self) -> None:
        """Reject non-JSON and incomplete successful API responses."""
        state = _decision_input()
        for response in (
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"answers": {}}),
        ):
            async def respond(_request: httpx.Request, result=response) -> httpx.Response:
                """Return the selected malformed response fixture."""
                return result

            provider = JevDecisionProvider(
                "key",
                "https://typesafe.example",
                "jev-test",
                0.1,
                transport=httpx.MockTransport(respond),
            )
            with self.assertRaises(InvalidDecisionError) as raised:
                await provider.decide(state)
            expected_reason = (
                "invalid_json" if response.text == "not json" else "agent_answer_missing"
            )
            self.assertEqual(raised.exception.reason, expected_reason)


class DecisionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_schedule_logs_and_skips_unavailable_configurations(self) -> None:
        """Skip shadow calls with a clear reason when config cannot call Jev."""
        cases = (
            (False, "configured", 1),
            (True, None, 1),
            (True, "configured", 0),
        )
        for enabled, api_key, max_calls in cases:
            provider = Mock(api_key=api_key)
            service = DecisionService(
                provider,
                DeterministicFallbackProvider(),
                enabled=enabled,
                min_confidence=0.7,
                max_calls_per_request=max_calls,
                input_cost_per_million_usd=0,
                output_cost_per_million_usd=0,
            )

            with self.subTest(enabled=enabled, api_key=bool(api_key), max_calls=max_calls):
                self.assertIsNone(service.schedule_shadow(_decision_input()))

    async def test_valid_decision_is_accepted_and_usage_is_preserved(self) -> None:
        """Accept an authorized provider decision and preserve token usage."""
        provider = AsyncMock()
        provider.decide.return_value = ProviderResult(
            decision=MidasDecision(
                agent="sustainability",
                tools=["get_consumption_summary"],
                needs_mcp=True,
                needs_analytics=False,
                confidence=0.92,
            ),
            model="jev-latest",
            input_tokens=30,
            output_tokens=4,
        )
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0.042,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input())

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.model, "jev-latest")
        self.assertEqual((outcome.input_tokens, outcome.output_tokens), (30, 4))

    async def test_disabled_service_does_not_call_provider(self) -> None:
        """Use fallback without a provider request when Jev is disabled."""
        provider = AsyncMock()
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=False,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input())

        self.assertEqual(outcome.fallback_reason, "disabled")
        provider.decide.assert_not_awaited()

    async def test_low_confidence_uses_fallback(self) -> None:
        """Reject a valid route whose confidence is below the configured floor."""
        provider = AsyncMock()
        provider.decide.return_value = ProviderResult(
            decision=MidasDecision(
                agent="sustainability",
                tools=["get_consumption_summary"],
                needs_mcp=True,
                needs_analytics=False,
                confidence=0.2,
            ),
            model="jev-latest",
            input_tokens=10,
            output_tokens=2,
        )
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input())

        self.assertEqual(outcome.fallback_reason, "low_confidence")
        self.assertEqual(outcome.decision.agent, "default")

    async def test_unavailable_tool_is_rejected_and_falls_back(self) -> None:
        """Reject tools that are outside the selected agent allowlist."""
        provider = AsyncMock()
        provider.decide.return_value = ProviderResult(
            decision=MidasDecision(
                agent="faq",
                tools=["get_consumption_summary"],
                needs_mcp=True,
                needs_analytics=False,
                confidence=0.99,
            ),
            model="jev-latest",
            input_tokens=10,
            output_tokens=2,
        )
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input())

        self.assertEqual(outcome.fallback_reason, "tool_not_allowed")
        self.assertEqual(outcome.decision.agent, "default")

    async def test_invalid_route_and_mcp_mismatch_have_specific_fallback_reasons(self) -> None:
        """Reject unavailable routes and inconsistent MCP strategy explicitly."""
        cases = (
            (
                MidasDecision(
                    agent="other",
                    needs_mcp=False,
                    needs_analytics=False,
                    confidence=0.99,
                ),
                "agent_unavailable",
            ),
            (
                MidasDecision(
                    agent="sustainability",
                    needs_mcp=True,
                    needs_analytics=False,
                    confidence=0.99,
                ),
                "mcp_strategy_mismatch",
            ),
        )
        for decision, reason in cases:
            provider = AsyncMock()
            provider.decide.return_value = ProviderResult(
                decision=decision,
                model="jev-latest",
                input_tokens=10,
                output_tokens=2,
            )
            service = DecisionService(
                provider,
                DeterministicFallbackProvider(),
                enabled=True,
                min_confidence=0.7,
                max_calls_per_request=1,
                input_cost_per_million_usd=0,
                output_cost_per_million_usd=0,
            )

            with self.subTest(reason=reason):
                outcome = await service.decide(_decision_input())
                self.assertEqual(outcome.fallback_reason, reason)

    async def test_call_limit_falls_back_without_calling_provider(self) -> None:
        """Enforce the per-request decision limit before calling the provider."""
        provider = AsyncMock()
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input(), call_number=2)

        self.assertEqual(outcome.fallback_reason, "call_limit")
        provider.decide.assert_not_awaited()

    async def test_provider_timeouts_and_unexpected_failures_fall_back(self) -> None:
        """Turn provider timeouts and unexpected exceptions into fallback."""
        for error, reason in (
            (DecisionTimeoutError("timed out"), "timeout"),
            (RuntimeError("unexpected"), "provider_error"),
        ):
            provider = AsyncMock()
            provider.decide.side_effect = error
            service = DecisionService(
                provider,
                DeterministicFallbackProvider(),
                enabled=True,
                min_confidence=0.7,
                max_calls_per_request=1,
                input_cost_per_million_usd=0,
                output_cost_per_million_usd=0,
            )

            with self.subTest(reason=reason):
                outcome = await service.decide(_decision_input())
                self.assertEqual(outcome.fallback_reason, reason)

    async def test_analytics_is_rejected_when_backend_is_unavailable(self) -> None:
        """Reject analytics strategy until an analytics backend is available."""
        provider = AsyncMock()
        provider.decide.return_value = ProviderResult(
            decision=MidasDecision(
                agent="sustainability",
                tools=[],
                needs_mcp=False,
                needs_analytics=True,
                confidence=0.95,
            ),
            model="jev-latest",
            input_tokens=10,
            output_tokens=2,
        )
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        outcome = await service.decide(_decision_input())

        self.assertEqual(outcome.fallback_reason, "analytics_unavailable")

    async def test_shadow_decision_records_agreement_after_completion(self) -> None:
        """Record shadow agreement after a valid decision finishes."""
        provider = AsyncMock()
        provider.api_key = "configured"
        provider.decide.return_value = ProviderResult(
            decision=MidasDecision(
                agent="sustainability",
                tools=[],
                needs_mcp=False,
                needs_analytics=False,
                confidence=0.95,
            ),
            model="jev-latest",
            input_tokens=10,
            output_tokens=2,
        )
        service = DecisionService(
            provider,
            DeterministicFallbackProvider(),
            enabled=True,
            min_confidence=0.7,
            max_calls_per_request=1,
            input_cost_per_million_usd=0,
            output_cost_per_million_usd=0,
        )

        task = service.schedule_shadow(_decision_input())
        self.assertIsNotNone(task)
        await task
        service.record_shadow_agreement(task, "sustainability")

        provider.decide.assert_awaited_once()


class JevRouterIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Save mutable Jev settings before each router test."""
        self.previous_enabled = settings.jev_enabled
        self.previous_shadow = settings.jev_shadow_mode

    def tearDown(self) -> None:
        """Restore Jev settings after each router test."""
        settings.jev_enabled = self.previous_enabled
        settings.jev_shadow_mode = self.previous_shadow

    async def test_active_jev_can_select_valid_route(self) -> None:
        """Apply a valid Jev route when shadow mode is disabled."""
        settings.jev_enabled = True
        settings.jev_shadow_mode = False
        decision = MidasDecision(
            agent="sustainability",
            tools=["get_consumption_summary"],
            needs_mcp=True,
            needs_analytics=False,
            confidence=0.92,
        )
        service = AsyncMock()
        service.decide.return_value = DecisionOutcome(decision=decision, model="jev")

        with (
            patch.object(graph, "_build_decision_input", return_value=_decision_input()),
            patch.object(graph, "_DECISION_SERVICE", service),
        ):
            result = await graph.route_request(
                {
                    "route": "",
                    "user_id": "private-user-id",
                    "messages": [HumanMessage(content="Pergunta ambígua")],
                    "input_guardrail": {"allowed": True},
                }
            )

        self.assertEqual(result["routes"], ["sustainability"])
        self.assertEqual(result["route_source"], "jev")
        self.assertEqual(result["decision_tools"], ["get_consumption_summary"])

    async def test_active_jev_routes_greetings_instead_of_using_fast_path(self) -> None:
        """Use Jev for greetings when active, matching the all-message policy."""
        settings.jev_enabled = True
        settings.jev_shadow_mode = False
        decision = MidasDecision(
            agent="faq",
            tools=[],
            needs_mcp=False,
            needs_analytics=False,
            confidence=0.92,
        )
        service = AsyncMock()
        service.decide.return_value = DecisionOutcome(decision=decision, model="jev")

        with (
            patch.object(graph, "_build_decision_input", return_value=_decision_input()),
            patch.object(graph, "_DECISION_SERVICE", service),
        ):
            result = await graph.route_request(
                {
                    "route": "",
                    "messages": [HumanMessage(content="Oi")],
                    "input_guardrail": {"allowed": True},
                }
            )

        self.assertEqual(result["routes"], ["faq"])
        self.assertEqual(result["route_source"], "jev")
        service.decide.assert_awaited_once()

    async def test_shadow_decision_preserves_existing_route(self) -> None:
        """Keep the existing deterministic route while Jev runs in shadow."""
        settings.jev_enabled = True
        settings.jev_shadow_mode = True
        service = Mock()
        service.schedule_shadow.return_value = None

        with (
            patch.object(graph, "_resolve_local_routes", return_value=(
                ["ranking"], "deterministic"
            )),
            patch.object(graph, "_build_decision_input", return_value=_decision_input()),
            patch.object(graph, "_DECISION_SERVICE", service),
        ):
            result = await graph.route_request(
                {
                    "route": "",
                    "messages": [HumanMessage(content="Indicador")],
                    "input_guardrail": {"allowed": True},
                }
            )

        self.assertEqual(result["routes"], ["ranking"])
        self.assertEqual(result["route_source"], "deterministic")
        service.decide.assert_not_called()

    async def test_guardrail_block_never_calls_jev(self) -> None:
        """Skip Jev entirely when the input guardrail blocks a request."""
        settings.jev_enabled = True
        service = AsyncMock()

        with patch.object(graph, "_DECISION_SERVICE", service):
            result = await graph.route_request(
                {
                    "route": "",
                    "messages": [HumanMessage(content="bloqueado")],
                    "input_guardrail": {"allowed": False},
                }
            )

        self.assertEqual(result["route_source"], "guardrail")
        service.decide.assert_not_awaited()
        service.schedule_shadow.assert_not_called()

    async def test_provider_failure_keeps_the_legacy_deterministic_route(self) -> None:
        """Retain the existing deterministic route when Jev falls back."""
        settings.jev_enabled = True
        settings.jev_shadow_mode = False
        service = AsyncMock()
        service.decide.return_value = DecisionOutcome(
            decision=MidasDecision(
                agent="default",
                tools=[],
                needs_mcp=False,
                needs_analytics=False,
                confidence=0,
            ),
            fallback_reason="timeout",
        )

        with (
            patch.object(graph, "_build_decision_input", return_value=_decision_input()),
            patch.object(graph, "_DECISION_SERVICE", service),
        ):
            result = await graph.route_request(
                {
                    "route": "",
                    "messages": [
                        HumanMessage(content="Como está meu consumo de água?")
                    ],
                    "input_guardrail": {"allowed": True},
                }
            )

        self.assertEqual(result["route_source"], "deterministic")
        self.assertEqual(result["routes"], ["sustainability"])


class JevToolSelectionTests(unittest.TestCase):
    def test_unselected_consumption_tool_is_not_passed_to_model(self) -> None:
        """Keep a mandatory prefetch tool out of model access when unselected."""
        consumption_tool = Mock()
        consumption_tool.name = "get_consumption_summary"
        knowledge_tool = Mock()
        knowledge_tool.name = "search_knowledge"

        available_for_model = _model_mcp_tools(
            [consumption_tool, knowledge_tool],
            prefetched_names=set(),
            selected_tools=set(),
        )

        self.assertEqual(available_for_model, [])

    def test_selected_tools_remain_available_unless_already_prefetched(self) -> None:
        """Expose selected tools to the model except tools already prefetched."""
        consumption_tool = Mock()
        consumption_tool.name = "get_consumption_summary"
        knowledge_tool = Mock()
        knowledge_tool.name = "search_knowledge"

        available_for_model = _model_mcp_tools(
            [consumption_tool, knowledge_tool],
            prefetched_names={"get_consumption_summary"},
            selected_tools={"get_consumption_summary", "search_knowledge"},
        )

        self.assertEqual(available_for_model, [knowledge_tool])
