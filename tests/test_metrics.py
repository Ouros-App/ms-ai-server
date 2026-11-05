import unittest
from types import SimpleNamespace

from prometheus_client import generate_latest

from app.core.metrics import (
    LLM_COST_USD,
    LLM_FALLBACKS,
    LLM_REQUESTS,
    LLM_TOKENS,
    LLMUsageCallback,
    _safe_route_source,
    observe_chat_result,
    observe_chat_routing,
)


class MetricsTest(unittest.TestCase):
    def test_chat_result_records_agents_and_tools(self) -> None:
        observe_chat_result(
            "success",
            ["router", "faq"],
            ["recall_user_memories"],
        )

    def test_chat_routing_bounds_route_source_labels(self) -> None:
        observe_chat_routing(
            ["sustainability"],
            "pending",
            ["sustainability"],
        )

        self.assertEqual(_safe_route_source("pending"), "pending")
        self.assertEqual(_safe_route_source("user-controlled-value"), "unknown")

    def test_llm_usage_tracks_openai_cost_and_provider_fallbacks(self) -> None:
        cost = LLM_COST_USD.labels("openai", "fast")
        input_tokens = LLM_TOKENS.labels("openai", "fast", "input")
        cost_before = cost._value.get()
        input_before = input_tokens._value.get()
        callback = LLMUsageCallback(
            provider="openai",
            profile="fast",
            is_fallback=False,
            input_cost_per_million_usd=0.10,
            output_cost_per_million_usd=0.50,
            cached_input_cost_per_million_usd=0.01,
        )
        response = SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata={
                                "input_tokens": 1_000,
                                "output_tokens": 1_000,
                                "input_token_details": {
                                    "cache_read": 200,
                                    "cache_creation": 100,
                                },
                            }
                        )
                    )
                ]
            ],
            llm_output=None,
        )

        request_count = LLM_REQUESTS.labels("openai", "fast", "success")
        requests_before = request_count._value.get()
        callback.on_llm_end(response)

        self.assertEqual(input_tokens._value.get() - input_before, 1_000)
        self.assertAlmostEqual(cost._value.get() - cost_before, 0.0005845)
        self.assertEqual(request_count._value.get() - requests_before, 1)

        exposition = generate_latest().decode()
        self.assertIn("ai_server_llm_tokens_total", exposition)
        self.assertIn("ai_server_llm_cost_usd_total", exposition)

        fallback_count = LLM_FALLBACKS.labels("groq", "fast")._value.get()
        fallback = LLMUsageCallback(provider="groq", profile="fast", is_fallback=True)
        fallback.on_llm_end(
            SimpleNamespace(
                generations=[],
                llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            )
        )
        self.assertEqual(LLM_FALLBACKS.labels("groq", "fast")._value.get() - fallback_count, 1)
