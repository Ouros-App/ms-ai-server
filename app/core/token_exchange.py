import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
SAFE_OAUTH_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "invalid_client",
        "invalid_grant",
        "unauthorized_client",
        "unsupported_grant_type",
        "invalid_scope",
        "invalid_target",
        "access_denied",
        "temporarily_unavailable",
        "server_error",
    }
)


class MCPTokenExchangeError(RuntimeError):
    """Raised when the backend cannot obtain a delegated MCP access token."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


def _oauth_error_code(response: httpx.Response) -> str:
    """Return an allowlisted OAuth error code without logging response content."""

    try:
        payload = response.json()
    except ValueError:
        return "unknown"
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if error in SAFE_OAUTH_ERROR_CODES else "unknown"


async def _exchange_with_client(
    subject_token: str,
    client: httpx.AsyncClient,
) -> str:
    """Exchange one validated user token for a downscoped Knowledge MCP token."""

    secret = settings.mcp_keycloak_token_exchange_client_secret
    if secret is None:
        logger.error(
            "mcp_token_exchange_failed reason=missing_client_secret client_id=%s audience=%s",
            settings.mcp_keycloak_token_exchange_client_id,
            settings.mcp_keycloak_token_exchange_audience,
        )
        raise MCPTokenExchangeError(
            "MCP token-exchange client secret is not configured",
            reason="missing_client_secret",
        )

    logger.info(
        "mcp_token_exchange_started client_id=%s audience=%s",
        settings.mcp_keycloak_token_exchange_client_id,
        settings.mcp_keycloak_token_exchange_audience,
    )

    try:
        response = await client.post(
            settings.effective_mcp_token_exchange_url,
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
    except httpx.TimeoutException as exc:
        logger.warning(
            "mcp_token_exchange_failed reason=timeout error=%s",
            type(exc).__name__,
        )
        raise MCPTokenExchangeError(
            "Keycloak token exchange timed out",
            reason="timeout",
        ) from exc
    except httpx.ConnectError as exc:
        logger.warning(
            "mcp_token_exchange_failed reason=connect_error error=%s",
            type(exc).__name__,
        )
        raise MCPTokenExchangeError(
            "Keycloak token exchange connection failed",
            reason="connect_error",
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "mcp_token_exchange_failed reason=http_error error=%s",
            type(exc).__name__,
        )
        raise MCPTokenExchangeError(
            "Keycloak token exchange is unavailable",
            reason="http_error",
        ) from exc

    if response.status_code != 200:
        oauth_error = _oauth_error_code(response)
        logger.warning(
            "mcp_token_exchange_failed reason=keycloak_rejected status=%s oauth_error=%s",
            response.status_code,
            oauth_error,
        )
        raise MCPTokenExchangeError(
            "Keycloak rejected the MCP token exchange",
            reason="keycloak_rejected",
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning(
            "mcp_token_exchange_failed reason=invalid_json status=%s",
            response.status_code,
        )
        raise MCPTokenExchangeError(
            "Keycloak returned an invalid token-exchange response",
            reason="invalid_json",
            status_code=response.status_code,
        ) from exc

    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(access_token, str) or not access_token:
        logger.warning(
            "mcp_token_exchange_failed reason=missing_access_token status=%s",
            response.status_code,
        )
        raise MCPTokenExchangeError(
            "Keycloak token-exchange response omitted access_token",
            reason="missing_access_token",
            status_code=response.status_code,
        )

    logger.info("mcp_token_exchange_succeeded status=%s", response.status_code)
    return access_token


async def exchange_mcp_access_token(subject_token: str) -> str:
    """Exchange a user token without exposing the confidential client to callers."""

    timeout = httpx.Timeout(settings.mcp_keycloak_token_exchange_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _exchange_with_client(subject_token, client)
