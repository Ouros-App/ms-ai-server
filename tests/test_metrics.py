import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.metrics import (
    _cached_input_tokens,
    _response_model_name,
    _safe_mcp_tool,
    _safe_model,
    _safe_route_source,
    _token_count,
    observe_chat_result,
    observe_chat_routing,
    observed_llm_ainvoke,
)


class MetricsTest(unittest.IsolatedAsyncioTestCase):
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

    def test_metric_labels_are_bounded(self) -> None:
        self.assertEqual(_safe_model("openai/gpt-oss-20b"), "openai/gpt-oss-20b")
        self.assertEqual(_safe_model("user-supplied-model"), "unknown")
        self.assertEqual(_safe_mcp_tool("get_consumption_summary"), "get_consumption_summary")
        self.assertEqual(_safe_mcp_tool("dynamic-user-value"), "unknown")

    def test_token_helpers_support_provider_metadata_shapes(self) -> None:
        response = SimpleNamespace(
            response_metadata={"model_name": "openai/gpt-oss-120b"},
        )
        self.assertEqual(_response_model_name(response), "openai/gpt-oss-120b")
        self.assertEqual(_token_count({"input_tokens": 120}, "input_tokens"), 120)
        self.assertEqual(
            _cached_input_tokens(
                {"input_token_details": {"cache_read": 35}}
            ),
            35,
        )

    async def test_observed_llm_invoke_records_cancellation_and_closes_gauge(self) -> None:
        model = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=asyncio.CancelledError())
        )
        request_counter = SimpleNamespace(inc=unittest.mock.Mock())
        duration_histogram = SimpleNamespace(observe=unittest.mock.Mock())

        with (
            patch(
                "app.core.metrics.LLM_REQUESTS.labels",
                return_value=request_counter,
            ) as requests,
            patch(
                "app.core.metrics.LLM_DURATION.labels",
                return_value=duration_histogram,
            ) as duration,
            patch("app.core.metrics.LLM_IN_FLIGHT.inc") as in_flight_inc,
            patch("app.core.metrics.LLM_IN_FLIGHT.dec") as in_flight_dec,
            self.assertRaises(asyncio.CancelledError),
        ):
            await observed_llm_ainvoke(model, [], "fast")

        requests.assert_called_once_with("fast", "unknown", "cancelled")
        duration.assert_called_once_with("fast", "unknown")
        request_counter.inc.assert_called_once_with()
        duration_histogram.observe.assert_called_once()
        in_flight_inc.assert_called_once_with()
        in_flight_dec.assert_called_once_with()

    async def test_observed_llm_invoke_returns_response_without_prompt_logging(self) -> None:
        response = SimpleNamespace(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 25,
                "input_token_details": {"cache_read": 20},
            },
            response_metadata={"model_name": "openai/gpt-oss-20b"},
        )
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        messages = [{"role": "user", "content": "private message"}]

        observed = await observed_llm_ainvoke(model, messages, "fast")

        self.assertIs(observed, response)
        model.ainvoke.assert_awaited_once_with(messages)


if __name__ == "__main__":
    unittest.main()
