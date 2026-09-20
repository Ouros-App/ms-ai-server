import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.agents.mcp import MCPToolProvider, forward_mcp_access_token
from app.core.auth import get_current_principal
from app.core.config import settings


class KeycloakAuthTests(unittest.IsolatedAsyncioTestCase):
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
        ):
            with self.assertRaises(HTTPException) as raised:
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
        ):
            with self.assertRaises(HTTPException) as raised:
                await get_current_principal(credentials)

        self.assertEqual(raised.exception.status_code, 401)

    async def test_validated_user_token_is_forwarded_to_mcp_provider(self) -> None:
        provider = MCPToolProvider(
            url="https://ms-midas-mcp.discloud.app/mcp/",
            access_token="legacy-service-token",
        )

        with forward_mcp_access_token("validated-user-token"):
            token = provider._token_for("42")

        self.assertEqual(token, "validated-user-token")


if __name__ == "__main__":
    unittest.main()
