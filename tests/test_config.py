import os
import unittest
from unittest.mock import patch

from app.agents.tools import TOOLS
from app.core.config import Settings, get_settings, settings


class ConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Settings(_env_file=None)

        self.assertEqual(config.app_port, 8000)
        self.assertEqual(config.mongodb_database, "mongodb-ai-prod")
        self.assertEqual(config.groq_model, "openai/gpt-oss-120b")
        self.assertEqual(config.groq_fast_model, "openai/gpt-oss-20b")
        self.assertEqual(config.nvidia_nim_model, "meta/llama-3.3-70b-instruct")
        self.assertEqual(config.nvidia_nim_fast_model, "meta/llama-3.1-8b-instruct")
        self.assertEqual(config.llm_total_timeout_seconds, 20)
        self.assertIsNone(config.auth_bearer_token)
        self.assertEqual(TOOLS, [])
        self.assertIs(get_settings(), settings)
