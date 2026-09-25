import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt.exceptions import PyJWKClientConnectionError

from app.agents.mcp import MCPToolProvider, forward_mcp_access_token
from app.core.auth import (
    _decode_keycloak_token,
    _jwks_url,
    _keycloak_principal,
    get_current_principal,
    user_id_for_request,
)
from app.core.config import settings


class KeycloakAuthTests(unittest.IsolatedAsyncioTestCase):
    def credentials(self, token: str = "signed-keycloak-token") -> HTTPAuthorizationCredentials:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    def test_jwks_url_supports_explicit_and_derived_modes(self) -> None:
        with (
            patch.object(settings, "auth_jwks_url", "https://keys.example/jwks"),
            patch.object(settings, "auth_jwt_issuer", "https://issuer.example"),
        ):
            self.assertEqual(_jwks_url(), "https://keys.example/jwks")

        with (
            patch.object(settings, "auth_jwks_url", None),
            patch.object(settings, "auth_jwt_issuer", "https://issuer.example/"),
        ):
            self.assertEqual(
                _jwks_url(),
                "https://issuer.example/protocol/openid-connect/certs",
            )

    def test_decode_keycloak_token_uses_rs256_contract(self) -> None:
        jwks_client = Mock()
        jwks_client.get_jwk_set.return_value = object()
        jwks_client.get_signing_key_from_jwt.return_value = SimpleNamespace(
            key="public-key"
        )
        claims = {
            "sub": "subject",
            "aud": ["ms-ai-server", "ms-ai-server-mcp-exchange"],
        }

        with (
            patch.object(settings, "auth_jwks_url", "https://keys.example/jwks"),
            patch.object(settings, "auth_jwt_issuer", "https://issuer.example"),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_client_id",
                "ms-ai-server-mcp-exchange",
            ),
            patch("app.core.auth._get_jwks_client", return_value=jwks_client),
            patch("app.core.auth.decode", return_value=claims) as decoder,
        ):
            self.assertEqual(_decode_keycloak_token("signed-token"), claims)

        jwks_client.get_jwk_set.assert_called_once_with()
        jwks_client.get_signing_key_from_jwt.assert_called_once_with("signed-token")
        self.assertEqual(decoder.call_args.kwargs["algorithms"], ["RS256"])
        self.assertEqual(decoder.call_args.kwargs["issuer"], "https://issuer.example")
        self.assertEqual(decoder.call_args.kwargs["audience"], "ms-ai-server")

    def test_decode_rejects_token_without_exchange_requester_audience(self) -> None:
        jwks_client = Mock()
        jwks_client.get_jwk_set.return_value = object()
        jwks_client.get_signing_key_from_jwt.return_value = SimpleNamespace(
            key="public-key"
        )
        claims = {
            "sub": "subject",
            "aud": ["ms-ai-server"],
        }

        with (
            patch.object(settings, "auth_jwks_url", "https://keys.example/jwks"),
            patch.object(settings, "auth_jwt_issuer", "https://issuer.example"),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch.object(
                settings,
                "mcp_keycloak_token_exchange_client_id",
                "ms-ai-server-mcp-exchange",
            ),
            patch("app.core.auth._get_jwks_client", return_value=jwks_client),
            patch("app.core.auth.decode", return_value=claims),
        ):
            self.assertIsNone(_decode_keycloak_token("signed-token"))

    async def test_jwks_connection_failure_is_service_unavailable(self) -> None:
        credentials = self.credentials()
        with (
            patch(
                "app.core.auth._decode_keycloak_token",
                side_effect=PyJWKClientConnectionError("jwks unavailable"),
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await get_current_principal(credentials)

        self.assertEqual(raised.exception.status_code, 503)

    async def test_valid_keycloak_claims_map_business_identity(self) -> None:
        claims = {
            "sub": "keycloak-subject",
            "database_id": 42,
            "account_type": "farm_owner",
            "realm_access": {"roles": ["farm_owner"]},
            "exp": 4_102_444_800,
        }
        with patch("app.core.auth._decode_keycloak_token", return_value=claims):
            principal = await get_current_principal(self.credentials())

        self.assertEqual(principal.subject, "keycloak-subject")
        self.assertEqual(principal.user_id, "42")
        self.assertEqual(principal.user_type, "farm_owner")
        self.assertEqual(principal.access_token, "signed-keycloak-token")
        self.assertEqual(principal.expires_at, 4_102_444_800.0)

    def test_business_identity_requires_matching_role_and_database_id(self) -> None:
        invalid_claims = (
            {
                "sub": "subject",
                "database_id": 42,
                "account_type": "admin",
                "realm_access": {"roles": ["farm_owner"]},
            },
            {
                "sub": "subject",
                "database_id": True,
                "account_type": "farm_owner",
                "realm_access": {"roles": ["farm_owner"]},
            },
            {
                "sub": "subject",
                "database_id": 0,
                "account_type": "farm_owner",
                "realm_access": {"roles": ["farm_owner"]},
            },
        )
        for claims in invalid_claims:
            with self.subTest(claims=claims):
                self.assertIsNone(_keycloak_principal(claims, "token"))

    async def test_invalid_keycloak_token_has_no_legacy_fallback(self) -> None:
        credentials = self.credentials("legacy-token")
        with (
            patch("app.core.auth._decode_keycloak_token", return_value=None),
            self.assertRaises(HTTPException) as raised,
        ):
            await get_current_principal(credentials)

        self.assertEqual(raised.exception.status_code, 401)

    async def test_user_id_is_derived_from_signed_claim(self) -> None:
        claims = {
            "sub": "subject",
            "database_id": 42,
            "account_type": "company_employee",
            "realm_access": {"roles": ["company_employee"]},
        }
        with patch("app.core.auth._decode_keycloak_token", return_value=claims):
            principal = await get_current_principal(self.credentials())

        self.assertEqual(user_id_for_request(None, principal), "42")
        self.assertEqual(user_id_for_request("42", principal), "42")
        with self.assertRaises(HTTPException):
            user_id_for_request("99", principal)

    async def test_validated_user_token_is_forwarded_to_mcp_only_in_request_context(self) -> None:
        provider = MCPToolProvider(url="https://mcp.example/mcp")

        with forward_mcp_access_token("validated-user-token"):
            self.assertEqual(provider._token_for(), "validated-user-token")

        self.assertIsNone(provider._token_for())
        self.assertNotEqual(
            provider._token_cache_key("validated-user-token"),
            "validated-user-token",
        )


if __name__ == "__main__":
    unittest.main()
