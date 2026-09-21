from pathlib import Path
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
STATIC_DIR = Path(__file__).with_name("static")

router = APIRouter(prefix="/debug", include_in_schema=False)


def _require_enabled() -> None:
    if not settings.debug_ui_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


async def _debug_principal(
    token: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> Principal:
    _require_enabled()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return await principal_from_token(token)


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


@router.post("/api/login", response_model=DebugSessionResponse)
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
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("missing access token")
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

    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        max_age=max(1, expires_in),
        httponly=True,
        secure=settings.debug_ui_cookie_secure,
        samesite="strict",
        path="/debug",
    )
    return _session(principal)


@router.post("/api/logout", status_code=status.HTTP_204_NO_CONTENT)
async def debug_logout(response: Response) -> None:
    _require_enabled()
    response.delete_cookie(
        COOKIE_NAME,
        path="/debug",
        secure=settings.debug_ui_cookie_secure,
        httponly=True,
        samesite="strict",
    )


@router.get("/api/session", response_model=DebugSessionResponse)
async def debug_session(
    principal: Annotated[Principal, Depends(_debug_principal)],
) -> DebugSessionResponse:
    return _session(principal)


@router.post("/api/chat", response_model=DebugChatResponse)
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
