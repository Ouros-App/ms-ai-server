import unittest

from app.core.metrics import observe_chat_result


class MetricsTest(unittest.TestCase):
    def test_chat_result_records_agents_and_tools(self) -> None:
        observe_chat_result(
            "success",
            ["router", "faq"],
            ["recall_user_memories"],
        )
