import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class MCPTokenExchangeError(RuntimeError):
    """Raised when the backend cannot obtain a delegated MCP access token."""


def _oauth_error(response: httpx.Response) -> str | None:
    """Return only the OAuth error code, never the response body or tokens."""

    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("error")
    return value if isinstance(value, str) and value else None


async def _exchange_with_client(
    subject_token: str,
    client: httpx.AsyncClient,
) -> str:
    """Exchange one validated user token for a downscoped Knowledge MCP token."""

    secret = settings.mcp_keycloak_token_exchange_client_secret
    if secret is None:
        raise MCPTokenExchangeError("MCP token-exchange client secret is not configured")

    try:
        response = await client.post(
            settings.mcp_keycloak_token_exchange_url,
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token": subject_token,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "audience": settings.mcp_keycloak_token_exchange_audience,
            },
            auth=httpx.BasicAuth(
                settings.mcp_keycloak_token_exchange_client_id,
                secret.get_secret_value(),
            ),
        )
    except httpx.HTTPError as exc:
        raise MCPTokenExchangeError("Keycloak token exchange is unavailable") from exc

    if response.status_code != 200:
        logger.warning(
            "mcp_token_exchange_failed status=%s oauth_error=%s",
            response.status_code,
            _oauth_error(response),
        )
        raise MCPTokenExchangeError("Keycloak rejected the MCP token exchange")

    try:
        payload = response.json()
    except ValueError as exc:
        raise MCPTokenExchangeError("Keycloak returned an invalid token-exchange response") from exc

    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise MCPTokenExchangeError("Keycloak token-exchange response omitted access_token")
    return access_token


async def exchange_mcp_access_token(subject_token: str) -> str:
    """Exchange a user token without exposing the confidential client to callers."""

    timeout = httpx.Timeout(settings.mcp_keycloak_token_exchange_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _exchange_with_client(subject_token, client)
