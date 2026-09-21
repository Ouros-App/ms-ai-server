import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, PyJWKClient, decode
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError, PyJWKSetError

from app.core.config import settings

VALID_ACCOUNT_TYPES = {"farm_owner", "company_employee", "admin"}


@dataclass(frozen=True)
class Principal:
    subject: str
    user_id: str
    user_type: str
    access_token: str = field(default="", repr=False, compare=False)


bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationKeyServiceError(RuntimeError):
    """Raised when Keycloak JWKS cannot provide a usable key set."""


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token Bearer invalido.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=8)
def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url, cache_keys=True, lifespan=300)


def _jwks_url() -> str:
    return (
        settings.auth_jwks_url
        or settings.auth_jwt_issuer.rstrip("/")
        + "/protocol/openid-connect/certs"
    )


def _get_signing_key(token: str):
    client = _get_jwks_client(_jwks_url())
    try:
        client.get_jwk_set()
        return client.get_signing_key_from_jwt(token)
    except PyJWKClientConnectionError:
        raise
    except PyJWKSetError as exc:
        raise AuthenticationKeyServiceError("invalid JWKS key set") from exc
    except (InvalidTokenError, PyJWKClientError, ValueError, TypeError):
        return None


def _decode_keycloak_token(token: str) -> dict | None:
    signing_key = _get_signing_key(token)
    if signing_key is None:
        return None
    try:
        claims = decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.auth_jwt_issuer,
            audience=settings.auth_jwt_audience,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except InvalidTokenError:
        return None
    return claims if isinstance(claims, dict) else None


def _keycloak_principal(claims: dict, token: str) -> Principal | None:
    subject = claims.get("sub")
    database_id = claims.get("database_id")
    account_type = claims.get("account_type")
    realm_access = claims.get("realm_access")
    roles = realm_access.get("roles") if isinstance(realm_access, dict) else None

    if not isinstance(subject, str) or not subject.strip():
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
        subject=subject,
        user_id=str(numeric_database_id),
        user_type=account_type,
        access_token=token,
    )


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> Principal:
    """Validate the only supported user credential: a Keycloak access token."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        claims = await asyncio.to_thread(
            _decode_keycloak_token,
            credentials.credentials,
        )
    except (PyJWKClientConnectionError, AuthenticationKeyServiceError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servico de chaves de autenticacao indisponivel.",
        ) from error

    if claims is None:
        raise _unauthorized()

    principal = _keycloak_principal(claims, credentials.credentials)
    if principal is None:
        raise _unauthorized()
    return principal


def user_id_for_request(
    requested_user_id: str | None,
    principal: Principal,
) -> str:
    """Use the signed database_id; a supplied compatibility ID may only match."""

    if requested_user_id is not None and requested_user_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="O user_id nao corresponde ao usuario autenticado.",
        )
    return principal.user_id
