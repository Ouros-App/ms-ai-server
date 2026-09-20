import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.agents.mcp import MCPToolProvider, forward_mcp_access_token
from app.core.auth import (
    _decode_keycloak_token,
    _jwks_url,
    get_current_principal,
)
from app.core.config import settings


class KeycloakAuthTests(unittest.IsolatedAsyncioTestCase):
    def test_jwks_url_supports_explicit_derived_and_disabled_modes(self) -> None:
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

        with (
            patch.object(settings, "auth_jwks_url", None),
            patch.object(settings, "auth_jwt_issuer", None),
        ):
            self.assertIsNone(_jwks_url())

    def test_decode_keycloak_token_uses_jwks_contract(self) -> None:
        signing_key = SimpleNamespace(key="public-key")
        jwks_client = Mock()
        jwks_client.get_signing_key_from_jwt.return_value = signing_key
        expected_claims = {"sub": "subject"}

        with (
            patch.object(settings, "auth_jwks_url", "https://keys.example/jwks"),
            patch.object(settings, "auth_jwt_issuer", "https://issuer.example"),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch(
                "app.core.auth._get_jwks_client",
                return_value=jwks_client,
            ),
            patch(
                "app.core.auth.decode",
                return_value=expected_claims,
            ) as decoder,
        ):
            claims = _decode_keycloak_token("signed-token")

        self.assertEqual(claims, expected_claims)
        jwks_client.get_signing_key_from_jwt.assert_called_once_with(
            "signed-token"
        )
        decoder.assert_called_once()
        self.assertEqual(decoder.call_args.kwargs["algorithms"], ["RS256"])
        self.assertEqual(
            decoder.call_args.kwargs["issuer"],
            "https://issuer.example",
        )
        self.assertEqual(
            decoder.call_args.kwargs["audience"],
            "ms-ai-server",
        )

    def test_decode_keycloak_token_requires_complete_configuration(self) -> None:
        with (
            patch.object(settings, "auth_jwks_url", None),
            patch.object(settings, "auth_jwt_issuer", None),
            patch.object(settings, "auth_jwt_audience", None),
        ):
            self.assertIsNone(_decode_keycloak_token("token"))

    async def test_keycloak_claims_map_business_identity(self) -> None:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="signed-keycloak-token",
        )
        claims = {
            "sub": "keycloak-subject",
            "database_id": 42,
            "account_type": "farm_owner",
            "realm_access": {"roles": ["farm_owner"]},
        }

        with (
            patch.object(
                settings,
                "auth_jwt_issuer",
                "https://ouros-keycloak.discloud.app/realms/ouros",
            ),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch.object(settings, "auth_bearer_token", None),
            patch.object(settings, "auth_jwt_secret", None),
            patch("app.core.auth._decode_keycloak_token", return_value=claims),
        ):
            principal = await get_current_principal(credentials)

        self.assertEqual(principal.subject, "keycloak-subject")
        self.assertEqual(principal.user_id, "42")
        self.assertEqual(principal.user_type, "farm_owner")
        self.assertEqual(principal.access_token, "signed-keycloak-token")

    async def test_keycloak_claim_requires_matching_realm_role(self) -> None:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="signed-keycloak-token",
        )
        claims = {
            "sub": "keycloak-subject",
            "database_id": 42,
            "account_type": "admin",
            "realm_access": {"roles": ["farm_owner"]},
        }

        with (
            patch.object(
                settings,
                "auth_jwt_issuer",
                "https://ouros-keycloak.discloud.app/realms/ouros",
            ),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch.object(settings, "auth_bearer_token", None),
            patch.object(settings, "auth_jwt_secret", None),
            patch("app.core.auth._decode_keycloak_token", return_value=claims),
            self.assertRaises(HTTPException) as raised,
        ):
            await get_current_principal(credentials)

        self.assertEqual(raised.exception.status_code, 401)

    async def test_keycloak_identity_does_not_fall_back_to_subject_as_database_id(
        self,
    ) -> None:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="signed-keycloak-token",
        )
        claims = {
            "sub": "keycloak-subject",
            "account_type": "farm_owner",
            "realm_access": {"roles": ["farm_owner"]},
        }

        with (
            patch.object(
                settings,
                "auth_jwt_issuer",
                "https://ouros-keycloak.discloud.app/realms/ouros",
            ),
            patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
            patch.object(settings, "auth_bearer_token", None),
            patch.object(settings, "auth_jwt_secret", None),
            patch("app.core.auth._decode_keycloak_token", return_value=claims),
            self.assertRaises(HTTPException) as raised,
        ):
            await get_current_principal(credentials)

        self.assertEqual(raised.exception.status_code, 401)

    async def test_keycloak_claim_rejects_malformed_identity_types(self) -> None:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="signed-keycloak-token",
        )
        malformed_claims = [
            {
                "sub": "subject",
                "database_id": True,
                "account_type": "farm_owner",
                "realm_access": {"roles": ["farm_owner"]},
            },
            {
                "sub": "subject",
                "database_id": 42.5,
                "account_type": "farm_owner",
                "realm_access": {"roles": ["farm_owner"]},
            },
            {
                "sub": "subject",
                "database_id": 42,
                "account_type": "farm_owner",
                "realm_access": {"roles": "farm_owner"},
            },
        ]

        for claims in malformed_claims:
            with (
                self.subTest(claims=claims),
                patch.object(
                    settings,
                    "auth_jwt_issuer",
                    "https://ouros-keycloak.discloud.app/realms/ouros",
                ),
                patch.object(settings, "auth_jwt_audience", "ms-ai-server"),
                patch.object(settings, "auth_bearer_token", None),
                patch.object(settings, "auth_jwt_secret", None),
                patch(
                    "app.core.auth._decode_keycloak_token",
                    return_value=claims,
                ),
                self.assertRaises(HTTPException),
            ):
                await get_current_principal(credentials)

    async def test_validated_user_token_is_forwarded_to_mcp_provider(self) -> None:
        provider = MCPToolProvider(
            url="https://ms-midas-mcp.discloud.app/mcp/",
            access_token="legacy-service-token",
        )

        with forward_mcp_access_token(
            "validated-user-token",
            "company_employee",
        ):
            token = provider._token_for("42")

        self.assertEqual(token, "validated-user-token")
        self.assertEqual(provider._token_for("42"), "legacy-service-token")
        self.assertNotEqual(
            provider._token_cache_key("validated-user-token"),
            "validated-user-token",
        )


if __name__ == "__main__":
    unittest.main()
