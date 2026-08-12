import importlib
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

from langgraph.checkpoint.memory import InMemorySaver


class MainTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_uses_mongodb_saver(self) -> None:
        context = MagicMock()
        context.__enter__.return_value = InMemorySaver()
        saver = Mock(from_conn_string=Mock(return_value=context))
        previous = sys.modules.pop("app.main", None)

        try:
            with patch.dict(sys.modules, {"langgraph.checkpoint.mongodb": Mock(MongoDBSaver=saver)}):
                main = importlib.import_module("app.main")
                async with main.lifespan(main.app):
                    self.assertIsNotNone(main.app.state.graph)
        finally:
            sys.modules.pop("app.main", None)
            if previous:
                sys.modules["app.main"] = previous

        saver.from_conn_string.assert_called_once()
