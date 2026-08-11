import unittest

from app.agents.tools import TOOLS
from app.core.config import Settings, get_settings, settings


class ConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        config = Settings()

        self.assertEqual(config.app_port, 8000)
        self.assertEqual(config.mongodb_database, "ai_server")
        self.assertEqual(TOOLS, [])
        self.assertIs(get_settings(), settings)
