import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, PyJWKClient, decode
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

from app.core.config import settings

VALID_ACCOUNT_TYPES = {"farm_owner", "company_employee", "admin"}


@dataclass(frozen=True)
class Principal:
    subject: str
    user_id: str | None = None
    user_type: str | None = None
    access_token: str = field(default="", repr=False, compare=False)
    forward_to_mcp: bool = False


bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token Bearer invalido.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=8)
def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Reuse the JWKS client so signing keys remain cached between requests."""

    return PyJWKClient(jwks_url, cache_keys=True, lifespan=300)


def _jwks_url() -> str | None:
    """Resolve the explicit JWKS URL or derive it from the configured issuer."""

    if settings.auth_jwks_url:
        return settings.auth_jwks_url
    if not settings.auth_jwt_issuer:
        return None
    return (
        settings.auth_jwt_issuer.rstrip("/")
        + "/protocol/openid-connect/certs"
    )


def _decode_keycloak_token(token: str) -> dict | None:
    """Validate a Keycloak access token using JWKS, issuer and audience."""

    issuer = settings.auth_jwt_issuer
    audience = settings.auth_jwt_audience
    jwks_url = _jwks_url()
    if not issuer or not audience or not jwks_url:
        return None

    try:
        signing_key = _get_jwks_client(jwks_url).get_signing_key_from_jwt(token)
        claims = decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            audience=audience,
            options={
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )
    except PyJWKClientConnectionError:
        raise
    except (InvalidTokenError, PyJWKClientError, ValueError):
        return None
    return claims if isinstance(claims, dict) else None


def _keycloak_principal(claims: dict, token: str) -> Principal | None:
    """Map signed Ouros identity claims to the API principal."""

    subject = claims.get("sub")
    database_id = claims.get("database_id")
    account_type = claims.get("account_type")
    realm_access = claims.get("realm_access")
    roles = realm_access.get("roles") if isinstance(realm_access, dict) else None

    if not isinstance(subject, (str, int)) or not str(subject).strip():
        return None
    if (
        not isinstance(account_type, str)
        or account_type not in VALID_ACCOUNT_TYPES
        or not isinstance(roles, list)
        or not all(isinstance(role, str) for role in roles)
        or account_type not in roles
    ):
        return None

    if isinstance(database_id, bool):
        return None
    if isinstance(database_id, int):
        numeric_database_id = database_id
    elif (
        isinstance(database_id, str)
        and database_id.isascii()
        and database_id.isdecimal()
    ):
        numeric_database_id = int(database_id)
    else:
        return None
    if numeric_database_id <= 0:
        return None

    return Principal(
        subject=str(subject),
        user_id=str(numeric_database_id),
        user_type=account_type,
        access_token=token,
        forward_to_mcp=True,
    )


def _legacy_hs256_principal(token: str) -> Principal | None:
    """Keep the previous HS256 flow available during the migration window."""

    jwt_secret = (
        settings.auth_jwt_secret.get_secret_value()
        if settings.auth_jwt_secret
        else ""
    )
    if not jwt_secret:
        return None

    options = {
        "verify_iss": bool(settings.auth_jwt_issuer),
        "verify_aud": bool(settings.auth_jwt_audience),
        "require": ["exp"],
    }
    try:
        claims = decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            issuer=settings.auth_jwt_issuer,
            audience=settings.auth_jwt_audience,
            options=options,
        )
    except InvalidTokenError:
        return None
    if not isinstance(claims, dict):
        return None

    subject = claims.get("sub") or claims.get("user_id")
    user_id = claims.get("user_id") or subject
    if (
        not isinstance(subject, (str, int))
        or not str(subject).strip()
        or not isinstance(user_id, (str, int))
        or not str(user_id).strip()
    ):
        return None
    user_type = claims.get("user_type")
    return Principal(
        subject=str(subject),
        user_id=str(user_id),
        user_type=user_type if isinstance(user_type, str) else None,
        access_token=token,
    )


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> Principal:
    """Validate Keycloak JWTs first, with legacy Bearer fallback for rollout."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    bearer_token = credentials.credentials
    expected_token = (
        settings.auth_bearer_token.get_secret_value()
        if settings.auth_bearer_token
        else ""
    )
    keycloak_configured = bool(
        settings.auth_jwt_issuer and settings.auth_jwt_audience
    )

    if not expected_token and not settings.auth_jwt_secret and not keycloak_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Autenticacao nao configurada.",
        )

    if keycloak_configured:
        try:
            claims = await asyncio.to_thread(
                _decode_keycloak_token,
                bearer_token,
            )
        except PyJWKClientConnectionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Servico de chaves de autenticacao indisponivel.",
            ) from error
        if claims is not None:
            principal = _keycloak_principal(claims, bearer_token)
            if principal is not None:
                return principal

    legacy_principal = _legacy_hs256_principal(bearer_token)
    if legacy_principal is not None:
        return legacy_principal

    if not expected_token or not compare_digest(bearer_token, expected_token):
        raise _unauthorized()

    return Principal(subject="shared-client", access_token=bearer_token)


def user_id_for_request(requested_user_id: str, principal: Principal) -> str:
    """Return only the identity authorized for personalized data."""

    if principal.user_id is None:
        if settings.auth_require_user_jwt:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Use um token de usuario para acessar dados personalizados.",
            )
        return requested_user_id
    if requested_user_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="O user_id nao corresponde ao usuario autenticado.",
        )
    return principal.user_id
