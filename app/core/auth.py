from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, decode

from app.core.config import settings


@dataclass(frozen=True)
class Principal:
    subject: str
    user_id: str | None = None
    user_type: str | None = None


bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token Bearer invalido.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> Principal:
    """Valida o token Bearer compartilhado e cria o principal da request."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    bearer_token = credentials.credentials
    expected_token = (
        settings.auth_bearer_token.get_secret_value() if settings.auth_bearer_token else ""
    )
    jwt_secret = (
        settings.auth_jwt_secret.get_secret_value() if settings.auth_jwt_secret else ""
    )
    if not expected_token and not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Autenticacao nao configurada.",
        )

    if jwt_secret:
        options = {
            "verify_iss": bool(settings.auth_jwt_issuer),
            "verify_aud": bool(settings.auth_jwt_audience),
        }
        try:
            claims = decode(
                bearer_token,
                jwt_secret,
                algorithms=["HS256"],
                issuer=settings.auth_jwt_issuer,
                audience=settings.auth_jwt_audience,
                options=options,
            )
        except InvalidTokenError:
            claims = None
        if isinstance(claims, dict):
            subject = claims.get("sub") or claims.get("user_id")
            user_id = claims.get("user_id") or subject
            if (
                isinstance(subject, (str, int))
                and str(subject).strip()
                and isinstance(user_id, (str, int))
                and str(user_id).strip()
            ):
                user_type = claims.get("user_type")
                return Principal(
                    subject=str(subject),
                    user_id=str(user_id),
                    user_type=user_type if isinstance(user_type, str) else None,
                )

    if not expected_token or not compare_digest(bearer_token, expected_token):
        raise _unauthorized()

    return Principal(subject="shared-client")


def user_id_for_request(requested_user_id: str, principal: Principal) -> str:
    """Retorna somente a identidade autorizada para dados personalizados."""
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
