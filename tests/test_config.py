import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

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
        self.assertEqual(config.nvidia_nim_model, "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(config.nvidia_nim_fast_model, "nvidia/nemotron-3-nano-30b-a3b")
        self.assertEqual(config.llm_total_timeout_seconds, 60)
        self.assertIsNone(config.auth_bearer_token)
        self.assertFalse(config.auth_require_user_jwt)
        self.assertEqual(TOOLS, [])
        self.assertIs(get_settings(), settings)

    def test_partial_keycloak_config_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                auth_jwt_issuer="https://ouros-keycloak.discloud.app/realms/ouros",
                auth_jwt_audience=None,
            )

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                auth_jwks_url="https://ouros-keycloak.discloud.app/realms/ouros/protocol/openid-connect/certs",
            )
