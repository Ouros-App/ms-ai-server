import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.agents.tools import TOOLS
from app.core.config import Settings, get_settings, settings


class ConfigTest(unittest.TestCase):
    def test_defaults_are_keycloak_only(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Settings(_env_file=None)

        self.assertEqual(config.app_port, 8000)
        self.assertEqual(config.mongodb_database, "mongodb-ai-prod")
        self.assertEqual(config.auth_jwt_issuer, "https://ouros-keycloak.discloud.app/realms/ouros")
        self.assertEqual(config.auth_jwt_audience, "ms-ai-server")
        self.assertIsNone(config.auth_jwks_url)
        self.assertEqual(
            config.mcp_url,
            "https://ms-midas-mcp.discloud.app/mcp/",
        )
        self.assertEqual(
            config.mcp_resource_url,
            "https://ms-midas-mcp.discloud.app/mcp/",
        )
        self.assertEqual(config.mcp_tools_cache_ttl_seconds, 300)
        self.assertEqual(config.debug_ui_keycloak_client_id, "ms-ai-server-debug")
        self.assertIsNone(config.debug_ui_keycloak_client_secret)
        self.assertEqual(TOOLS, [])
        self.assertIs(get_settings(), settings)

    def test_blank_keycloak_contract_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                auth_jwt_issuer="",
                auth_jwt_audience="ms-ai-server",
            )

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                auth_jwt_issuer="https://issuer.example",
                auth_jwt_audience="",
            )

    def test_debug_ui_requires_confidential_keycloak_client_secret(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                debug_ui_enabled=True,
                debug_ui_keycloak_client_secret=None,
            )

        with patch.dict(os.environ, {}, clear=True):
            config = Settings(
                _env_file=None,
                debug_ui_enabled=True,
                debug_ui_keycloak_client_secret="test-value",
            )

        self.assertEqual(
            config.debug_ui_keycloak_client_secret.get_secret_value(),
            "test-value",
        )


if __name__ == "__main__":
    unittest.main()
