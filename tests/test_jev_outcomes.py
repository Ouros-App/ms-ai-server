import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.decision.jev_provider import InvalidDecisionError, JevDecisionProvider
from app.decision.outcomes import JevOutcomeService
from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph


class JevOutcomeProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_outcome_request_contains_only_scrubbed_task_text(self) -> None:
        captured: dict[str, object] = {}

        async def respond(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "outcome": {
                            "type": "choice",
                            "choice": "resolved",
                            "confidence": 0.92,
                        }
                    },
                    "usage": {"input_tokens": 20, "output_tokens": 3},
                    "model": "jev-test",
                },
            )

        provider = JevDecisionProvider(
            "test-key",
            "https://typesafe.example",
            "jev-test",
            1,
            transport=httpx.MockTransport(respond),
        )
        result = await provider.classify_outcome(
            "consulte os dados de [PII_EMAIL_abc]",
            "A consulta foi concluída.",
        )

        self.assertEqual(result.outcome, "resolved")
        self.assertEqual(result.confidence, 0.92)
        self.assertEqual((result.input_tokens, result.output_tokens), (20, 3))
        payload = captured["payload"]
        self.assertEqual(
            payload["state"],
            {
                "request": "consulte os dados de [PII_EMAIL_abc]",
                "response": "A consulta foi concluída.",
            },
        )
        self.assertEqual(
            set(payload["questions"]["outcome"]["criteria"]), {"resolved", "unresolved"}
        )

    def test_invalid_outcome_choice_is_rejected(self) -> None:
        with self.assertRaises(InvalidDecisionError):
            JevDecisionProvider._parse_outcome(
                {
                    "answers": {
                        "outcome": {
                            "type": "choice",
                            "choice": "maybe",
                            "confidence": 0.9,
                        }
                    }
                }
            )


class JevOutcomeServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_scrubs_pii_and_respects_call_budget(self) -> None:
        provider = SimpleNamespace(
            api_key="configured",
            classify_outcome=AsyncMock(
                return_value=SimpleNamespace(
                    outcome="resolved",
                    confidence=0.9,
                    model="jev-test",
                    input_tokens=10,
                    output_tokens=2,
                )
            ),
        )
        service = JevOutcomeService(provider)
        with (
            patch.object(settings, "jev_enabled", True),
            patch.object(settings, "jev_shadow_mode", False),
            patch.object(settings, "jev_max_calls_per_request", 2),
            patch.object(settings, "jev_min_confidence", 0.7),
            patch.object(settings, "jev_input_cost_per_million_usd", 0.1),
            patch.object(settings, "jev_output_cost_per_million_usd", 0.2),
            patch("app.decision.outcomes.observe_jev_outcome") as observe,
        ):
            service.schedule(
                "Consulte pessoa@example.com",
                "A resposta menciona pessoa@example.com.",
                decision_calls=1,
            )
            tasks = tuple(service._tasks)
            await asyncio.gather(*tasks)

            provider.classify_outcome.assert_awaited_once()
            request, response = provider.classify_outcome.await_args.args
            self.assertNotIn("pessoa@example.com", request)
            self.assertNotIn("pessoa@example.com", response)
            observe.assert_called_once()
            self.assertEqual(observe.call_args.args[0], "resolved")
            self.assertEqual(observe.call_args.kwargs["input_tokens"], 10)
            self.assertEqual(observe.call_args.kwargs["output_tokens"], 2)
            self.assertAlmostEqual(observe.call_args.kwargs["cost_usd"], 0.0000014)

            provider.classify_outcome.return_value = SimpleNamespace(
                outcome="resolved",
                confidence=0.5,
                model="jev-test",
                input_tokens=10,
                output_tokens=2,
            )
            await service._classify("question", "response")
            self.assertEqual(observe.call_args.args[0], "uncertain")

            service.schedule("question", "answer", decision_calls=2)
            self.assertEqual(len(service._tasks), 0)
            self.assertEqual(provider.classify_outcome.await_count, 2)


class ChatOutcomeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_chat_schedules_outcome_classification(self) -> None:
        graph = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values={})),
            ainvoke=AsyncMock(
                return_value={
                    "messages": [AIMessage(content="Tarefa concluída.")],
                    "agents": ["default"],
                    "tools": [],
                    "routes": ["default"],
                    "route_source": "deterministic",
                    "specialist_results": [],
                    "pending_routes": [],
                    "pending_missing_data": [],
                    "pending_by_route": {},
                    "pending_personal_routes": [],
                    "decision_calls": 1,
                }
            ),
        )
        guardrail = SimpleNamespace(
            allowed=True,
            category="APROVADO",
            message="",
            sanitized_text="pergunta anonimizada",
            as_state=lambda: {
                "allowed": True,
                "sanitized_text": "pergunta anonimizada",
            },
        )

        with (
            patch(
                "app.services.chat.guard_input",
                new=AsyncMock(return_value=guardrail),
            ),
            patch("app.services.chat.jev_outcome_service.schedule") as schedule,
        ):
            response = await invoke_graph(
                graph,
                ChatRequest(
                    user_id="user-1",
                    thread_id="thread-1",
                    message="pergunta original",
                ),
                "user-1",
            )

        self.assertEqual(response.message, "Tarefa concluída.")
        schedule.assert_called_once_with(
            "pergunta anonimizada",
            "Tarefa concluída.",
            decision_calls=1,
        )
