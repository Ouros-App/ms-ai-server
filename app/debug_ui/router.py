from pathlib import Path
from time import time
from typing import Annotated

import httpx
from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jwt import InvalidTokenError, decode as decode_jwt

from app.core.auth import Principal, principal_from_token
from app.core.config import settings
from app.debug_ui.schemas import (
    DebugChatRequest,
    DebugChatResponse,
    DebugLoginRequest,
    DebugSessionResponse,
)
from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph_debug

COOKIE_NAME = "ouros_debug_session"
REFRESH_COOKIE_NAME = "ouros_debug_refresh"
DEBUG_PREFIX = "/debug"
REFRESH_LEEWAY_SECONDS = 60
STATIC_DIR = Path(__file__).with_name("static")

router = APIRouter(prefix=DEBUG_PREFIX, include_in_schema=False)


def _require_enabled() -> None:
    if not settings.debug_ui_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


def _set_access_cookie(response: Response, token: str, expires_in: int) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=max(1, expires_in),
        httponly=True,
        secure=settings.debug_ui_cookie_secure,
        samesite="strict",
        path=DEBUG_PREFIX,
    )


def _set_refresh_cookie(
    response: Response,
    token: str,
    expires_in: int | None,
) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=max(1, expires_in or 1800),
        httponly=True,
        secure=settings.debug_ui_cookie_secure,
        samesite="strict",
        path=DEBUG_PREFIX,
    )


def _delete_session_cookies(response: Response) -> None:
    for cookie_name in (COOKIE_NAME, REFRESH_COOKIE_NAME):
        response.delete_cookie(
            cookie_name,
            path=DEBUG_PREFIX,
            secure=settings.debug_ui_cookie_secure,
            httponly=True,
            samesite="strict",
        )


def _token_needs_refresh(token: str) -> bool:
    """Use an already validated JWT expiry only as a proactive refresh hint."""
    try:
        claims = decode_jwt(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
    except (InvalidTokenError, TypeError, ValueError):
        return False
    expires_at = claims.get("exp")
    return (
        isinstance(expires_at, (int, float))
        and not isinstance(expires_at, bool)
        and expires_at <= time() + REFRESH_LEEWAY_SECONDS
    )


async def _refresh_debug_session(
    refresh_token: str,
    response: Response,
) -> Principal:
    refresh_url = (
        settings.debug_ui_auth_service_url.rstrip("/")
        + "/v1/auth/token/refresh"
    )
    try:
        async with httpx.AsyncClient(
            timeout=settings.debug_ui_request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            upstream = await client.post(
                refresh_url,
                json={"refresh_token": refresh_token},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service indisponível.",
        ) from exc

    if upstream.status_code == 401:
        _delete_session_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão expirada.",
        )
    if upstream.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Não foi possível renovar a sessão.",
        )

    try:
        document = upstream.json()
        access_token = document["access_token"]
        expires_in = int(document.get("expires_in", 600))
        rotated_refresh_token = document.get("refresh_token") or refresh_token
        refresh_expires_in = document.get("refresh_expires_in")
        if refresh_expires_in is not None:
            refresh_expires_in = int(refresh_expires_in)
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("missing access token")
        if not isinstance(rotated_refresh_token, str) or not rotated_refresh_token:
            raise ValueError("missing refresh token")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Auth service devolveu uma resposta de refresh inválida.",
        ) from exc

    try:
        principal = await principal_from_token(access_token)
    except HTTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Auth service devolveu um token renovado incompatível.",
        ) from exc

    _set_access_cookie(response, access_token, expires_in)
    _set_refresh_cookie(response, rotated_refresh_token, refresh_expires_in)
    return principal


async def _debug_principal(
    response: Response,
    token: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
    refresh_token: Annotated[
        str | None,
        Cookie(alias=REFRESH_COOKIE_NAME),
    ] = None,
) -> Principal:
    _require_enabled()

    if token:
        try:
            principal = await principal_from_token(token)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_401_UNAUTHORIZED:
                raise
        else:
            if not _token_needs_refresh(token):
                return principal

    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    return await _refresh_debug_session(refresh_token, response)


def _session(principal: Principal) -> DebugSessionResponse:
    return DebugSessionResponse(
        user_id=principal.user_id,
        account_type=principal.user_type,
    )


@router.get("")
@router.get("/")
async def debug_index() -> FileResponse:
    _require_enabled()
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/api/login")
async def debug_login(
    payload: DebugLoginRequest,
    response: Response,
) -> DebugSessionResponse:
    _require_enabled()
    token_url = (
        settings.debug_ui_auth_service_url.rstrip("/")
        + "/v1/auth/token"
    )
    try:
        async with httpx.AsyncClient(
            timeout=settings.debug_ui_request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            upstream = await client.post(
                token_url,
                json={
                    "email": payload.email,
                    "password": payload.password.get_secret_value(),
                },
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service indisponível.",
        ) from exc

    if upstream.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas.",
        )
    if upstream.status_code == 429:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Muitas tentativas. Tente novamente mais tarde.",
            headers={"Retry-After": upstream.headers.get("Retry-After", "60")},
        )
    if upstream.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login temporariamente indisponível.",
        )

    try:
        document = upstream.json()
        access_token = document["access_token"]
        expires_in = int(document.get("expires_in", 600))
        refresh_token = document.get("refresh_token")
        refresh_expires_in = document.get("refresh_expires_in")
        if refresh_expires_in is not None:
            refresh_expires_in = int(refresh_expires_in)
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("missing access token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ValueError("missing refresh token")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Auth service devolveu uma resposta inválida.",
        ) from exc

    try:
        principal = await principal_from_token(access_token)
    except HTTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Auth service devolveu um token incompatível com o AI Server.",
        ) from exc

    _set_access_cookie(response, access_token, expires_in)
    _set_refresh_cookie(response, refresh_token, refresh_expires_in)
    return _session(principal)


@router.post("/api/logout", status_code=status.HTTP_204_NO_CONTENT)
async def debug_logout(response: Response) -> None:
    _require_enabled()
    _delete_session_cookies(response)


@router.get("/api/session")
async def debug_session(
    principal: Annotated[Principal, Depends(_debug_principal)],
) -> DebugSessionResponse:
    return _session(principal)


@router.post("/api/chat")
async def debug_chat(
    payload: DebugChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(_debug_principal)],
) -> DebugChatResponse:
    response, diagnostics = await invoke_graph_debug(
        request.app.state.graph,
        ChatRequest(
            user_id=principal.user_id,
            message=payload.message,
            thread_id=payload.thread_id,
        ),
        principal.user_id,
        getattr(request.app.state, "thread_ownership", None),
        principal_token=principal.access_token,
    )
    return DebugChatResponse(
        thread_id=response.thread_id,
        message=response.message,
        agents=response.agents,
        tools=response.tools,
        **diagnostics,
    )


def install_debug_ui(app: FastAPI) -> None:
    """Mount the optional debug product without coupling it to normal routes."""

    if not settings.debug_ui_enabled:
        return
    app.include_router(router)
    app.mount(
        "/debug/assets",
        StaticFiles(directory=STATIC_DIR),
        name="debug-ui-assets",
    )
