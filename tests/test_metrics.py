import unittest

from app.core.metrics import (
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
