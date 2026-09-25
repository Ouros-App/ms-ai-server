import unittest
import urllib.parse
from unittest.mock import patch

import httpx
from pydantic import SecretStr

from app.core.config import settings
from app.core.token_exchange import (
    ACCESS_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT,
    MCPTokenExchangeError,
    _exchange_with_client,
)


class TokenExchangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exchange_uses_confidential_client_and_downscopes_audience(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("authorization", "")
            captured["body"] = request.content.decode()
            return httpx.Response(
                200,
                json={"access_token": "delegated-mcp-token", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(handler)
        with (
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_client_secret",
                SecretStr("exchange-secret"),
            ),
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_client_id",
                "ms-ai-server-mcp-exchange",
            ),
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_audience",
                "ms-mcp-server-ouros-knowledge",
            ),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                token = await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(token, "delegated-mcp-token")
        self.assertTrue(captured["authorization"].startswith("Basic "))
        form = urllib.parse.parse_qs(captured["body"])
        self.assertEqual(form["grant_type"], [TOKEN_EXCHANGE_GRANT])
        self.assertEqual(form["subject_token"], ["mobile-user-token"])
        self.assertEqual(form["subject_token_type"], [ACCESS_TOKEN_TYPE])
        self.assertEqual(form["requested_token_type"], [ACCESS_TOKEN_TYPE])
        self.assertEqual(
            form["audience"],
            ["ms-mcp-server-ouros-knowledge"],
        )

    async def test_exchange_rejects_non_200_without_falling_back(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                403,
                json={"error": "not_allowed", "error_description": "sensitive detail"},
            )
        )
        with patch.object(
            settings,
            "mcp_keycloak_token_exchange_client_secret",
            SecretStr("exchange-secret"),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(raised.exception.reason, "keycloak_rejected")
        self.assertEqual(raised.exception.status_code, 403)

    async def test_exchange_requires_backend_secret(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: self.fail("HTTP must not run without a client secret")
        )
        with patch.object(
            settings,
            "mcp_keycloak_token_exchange_client_secret",
            None,
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(raised.exception.reason, "missing_client_secret")
        self.assertIsNone(raised.exception.status_code)

    async def test_exchange_classifies_connect_errors_without_logging_credentials(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        transport = httpx.MockTransport(handler)
        with (
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_client_secret",
                SecretStr("super-secret-value"),
            ),
            self.assertLogs("app.core.token_exchange", level="WARNING") as logs,
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("user-token-material", client)

        self.assertEqual(raised.exception.reason, "connect_error")
        joined = "\n".join(logs.output)
        self.assertIn("reason=connect_error", joined)
        self.assertNotIn("super-secret-value", joined)
        self.assertNotIn("user-token-material", joined)

    async def test_exchange_classifies_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        transport = httpx.MockTransport(handler)
        with patch.object(
            settings,
            "mcp_keycloak_token_exchange_client_secret",
            SecretStr("exchange-secret"),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(raised.exception.reason, "timeout")

    async def test_exchange_rejects_invalid_json_response(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"not-json")
        )
        with patch.object(
            settings,
            "mcp_keycloak_token_exchange_client_secret",
            SecretStr("exchange-secret"),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(raised.exception.reason, "invalid_json")
        self.assertEqual(raised.exception.status_code, 200)

    async def test_exchange_rejects_success_without_access_token(self) -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"token_type": "Bearer"})
        )
        with patch.object(
            settings,
            "mcp_keycloak_token_exchange_client_secret",
            SecretStr("exchange-secret"),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(MCPTokenExchangeError) as raised:
                    await _exchange_with_client("mobile-user-token", client)

        self.assertEqual(raised.exception.reason, "missing_access_token")
        self.assertEqual(raised.exception.status_code, 200)


if __name__ == "__main__":
    unittest.main()
