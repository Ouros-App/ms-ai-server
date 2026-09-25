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
            config.effective_mcp_token_exchange_url,
            (
                "https://ouros-keycloak.discloud.app/realms/ouros"
                "/protocol/openid-connect/token"
            ),
        )
        self.assertEqual(
            config.effective_debug_ui_token_url,
            (
                "https://ouros-keycloak.discloud.app/realms/ouros"
                "/protocol/openid-connect/token"
            ),
        )
        self.assertEqual(config.mcp_tools_cache_ttl_seconds, 300)
        self.assertEqual(
            config.mcp_keycloak_token_exchange_client_id,
            "ms-ai-server-mcp-exchange",
        )
        self.assertEqual(
            config.mcp_keycloak_token_exchange_audience,
            "ms-mcp-server-ouros-knowledge",
        )
        self.assertIsNone(config.mcp_keycloak_token_exchange_client_secret)
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

    def test_configured_mcp_exchange_requires_complete_contract(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                mcp_keycloak_token_exchange_client_secret="secret",
                mcp_keycloak_token_exchange_client_id="",
            )

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                mcp_keycloak_token_exchange_client_secret="secret",
                mcp_keycloak_token_exchange_audience="",
            )

    def test_keycloak_token_endpoint_overrides_are_optional(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Settings(
                _env_file=None,
                auth_jwt_issuer="https://issuer.example/realms/ouros",
            )

            self.assertEqual(
                config.effective_mcp_token_exchange_url,
                "https://issuer.example/realms/ouros/protocol/openid-connect/token",
            )
            self.assertEqual(
                config.effective_debug_ui_token_url,
                "https://issuer.example/realms/ouros/protocol/openid-connect/token",
            )

            overridden = Settings(
                _env_file=None,
                mcp_keycloak_token_exchange_url="https://auth.example/token",
                debug_ui_keycloak_token_url="https://debug-auth.example/token",
            )
            self.assertEqual(
                overridden.effective_mcp_token_exchange_url,
                "https://auth.example/token",
            )
            self.assertEqual(
                overridden.effective_debug_ui_token_url,
                "https://debug-auth.example/token",
            )

    def test_mcp_exchange_timeout_must_be_positive(self) -> None:
        for timeout in (0, -1):
            with (
                self.subTest(timeout=timeout),
                patch.dict(os.environ, {}, clear=True),
                self.assertRaises(ValidationError),
            ):
                Settings(
                    _env_file=None,
                    mcp_keycloak_token_exchange_timeout_seconds=timeout,
                )

    def test_mcp_tool_timeout_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, mcp_tool_timeout_seconds=0)
        with self.assertRaises(ValueError):
            Settings(_env_file=None, mcp_tool_timeout_seconds=31)

    def test_debug_ui_requires_token_url_and_client_id(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                debug_ui_enabled=True,
                debug_ui_keycloak_token_url="",
                debug_ui_keycloak_client_secret="test-value",
            )

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings(
                _env_file=None,
                debug_ui_enabled=True,
                debug_ui_keycloak_client_id="",
                debug_ui_keycloak_client_secret="test-value",
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
