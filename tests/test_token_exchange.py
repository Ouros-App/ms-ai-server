import unittest
import urllib.parse
from unittest.mock import patch

import httpx
from pydantic import SecretStr

from app.core.config import settings
from app.core.token_exchange import (
    ACCESS_TOKEN_TYPE,
    MCPTokenExchangeError,
    TOKEN_EXCHANGE_GRANT,
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
                with self.assertRaises(MCPTokenExchangeError):
                    await _exchange_with_client("mobile-user-token", client)

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
                with self.assertRaises(MCPTokenExchangeError):
                    await _exchange_with_client("mobile-user-token", client)


if __name__ == "__main__":
    unittest.main()
