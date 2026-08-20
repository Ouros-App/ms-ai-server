from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings


@dataclass(frozen=True)
class Principal:
    subject: str


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

    expected_token = (
        settings.auth_bearer_token.get_secret_value() if settings.auth_bearer_token else ""
    )
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Autenticacao nao configurada.",
        )

    if not compare_digest(credentials.credentials, expected_token):
        raise _unauthorized()

    return Principal(subject="shared-client")
