from array import array
from collections.abc import Iterator
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from time import time
from typing import Annotated
from urllib.parse import parse_qs

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
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from app.core.auth import Principal, principal_from_token
from app.core.config import settings
from app.debug_ui.schemas import (
    DebugChatRequest,
    DebugChatResponse,
    DebugConversationListResponse,
    DebugLoginRequest,
    DebugSessionResponse,
)
from app.schemas.chat import ChatRequest
from app.schemas.history import HistoryResponse
from app.services.chat import invoke_graph_debug
from app.services.history import get_thread_history

COOKIE_NAME = "ouros_debug_session"
REFRESH_COOKIE_NAME = "ouros_debug_refresh"
DEBUG_PREFIX = "/debug"
REFRESH_LEEWAY_SECONDS = 60
MAX_DEBUG_VISUALIZATION_HTML_CHARS = 1_500_000
MAX_DEBUG_VISUALIZATION_FORM_BYTES = 6_000_000
STATIC_DIR = Path(__file__).with_name("static")

_SCRIPT_TYPES = {
    "text/javascript",
    "application/javascript",
    "text/ecmascript",
    "application/ecmascript",
}
_REGEX_PREFIX_TOKENS = {
    "",
    "(",
    "[",
    "{",
    "=",
    ":",
    ",",
    ";",
    "!",
    "?",
    "=>",
    "&&",
    "||",
    "??",
    "return",
    "case",
    "throw",
    "delete",
    "void",
    "typeof",
    "new",
    "in",
    "of",
    "yield",
    "await",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
    "<",
    ">",
    "==",
    "!=",
    "===",
    "!==",
    "<=",
    ">=",
    "**",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
}
_CONTROL_PAREN_TOKENS = {"if", "for", "while", "switch", "catch", "with"}
_METHOD_BOUNDARY_TOKENS = {
    "{", "}", ",", "static", "get", "set", "async", "*"
}


def _is_js_identifier_start(character: str) -> bool:
    return (
        character == "_"
        or character == "$"
        or character.isalpha()
        or ord(character) >= 128
    )


def _is_js_identifier_part(character: str) -> bool:
    return _is_js_identifier_start(character) or character.isdigit()


def _skip_js_quoted_literal(source: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(source):
        character = source[index]
        if character == "\\":
            index += 2
        elif character == quote:
            return index + 1
        else:
            index += 1
    return len(source)


def _skip_js_regex_literal(source: str, start: int) -> int:
    index = start + 1
    in_character_class = False
    while index < len(source):
        character = source[index]
        if character in "\r\n":
            return start + 1
        if character == "\\":
            index += 2
            continue
        if character == "[":
            in_character_class = True
        elif character == "]":
            in_character_class = False
        elif character == "/" and not in_character_class:
            index += 1
            while index < len(source) and _is_js_identifier_part(source[index]):
                index += 1
            return index
        index += 1
    return start + 1


def _skip_js_template_literal(source: str, start: int) -> int:
    frames: list[tuple[bool, int]] = [(True, 0)]
    previous_token = ""
    index = start + 1
    while index < len(source):
        in_template, brace_depth = frames[-1]
        character = source[index]
        if character == "\\":
            index += 2
        elif in_template and character == "`":
            frames.pop()
            index += 1
            if not frames:
                return index
            previous_token = "<literal>"
        elif in_template and source.startswith("${", index):
            frames[-1] = (False, 1)
            previous_token = ""
            index += 2
        elif in_template:
            index += 1
        elif character.isspace():
            index += 1
        elif source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            index = len(source) if closing < 0 else closing + 2
        elif character in "'\"":
            index = _skip_js_quoted_literal(source, index, character)
            previous_token = "<literal>"
        elif character == "`":
            frames.append((True, 0))
            previous_token = "<literal>"
            index += 1
        elif character == "/" and previous_token in _REGEX_PREFIX_TOKENS:
            regex_end = _skip_js_regex_literal(source, index)
            if regex_end == index + 1:
                previous_token = "/"
                index += 1
            else:
                previous_token = "<literal>"
                index = regex_end
        elif character == "{":
            frames[-1] = (False, brace_depth + 1)
            previous_token = character
            index += 1
        elif character == "}":
            brace_depth -= 1
            frames[-1] = (True, 0) if brace_depth == 0 else (False, brace_depth)
            previous_token = character
            index += 1
        elif _is_js_identifier_start(character):
            token_start = index
            index += 1
            while index < len(source) and _is_js_identifier_part(source[index]):
                index += 1
            previous_token = source[token_start:index]
        elif character.isdigit():
            index += 1
            while index < len(source) and (
                _is_js_identifier_part(source[index]) or source[index] == "."
            ):
                index += 1
            previous_token = "<literal>"
        else:
            previous_token = character
            index += 1
    return len(source)


def _iter_js_tokens(source: str) -> Iterator[tuple[str, str, int, int]]:
    index = 0
    previous_token = ""
    while index < len(source):
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            index = len(source) if closing < 0 else closing + 2
            continue
        if character in "'\"`":
            start = index
            index = (
                _skip_js_template_literal(source, index)
                if character == "`"
                else _skip_js_quoted_literal(source, index, character)
            )
            token, kind = "<literal>", "literal"
        elif character == "/" and previous_token in _REGEX_PREFIX_TOKENS:
            start = index
            index = _skip_js_regex_literal(source, index)
            if index == start + 1:
                index += 1
                token, kind = "/", "punctuation"
            else:
                token, kind = "<literal>", "literal"
        elif _is_js_identifier_start(character):
            start = index
            index += 1
            while index < len(source) and _is_js_identifier_part(source[index]):
                index += 1
            token, kind = source[start:index], "identifier"
        elif character.isdigit():
            start = index
            index += 1
            while index < len(source) and (
                _is_js_identifier_part(source[index]) or source[index] == "."
            ):
                index += 1
            token, kind = "<literal>", "literal"
        else:
            start = index
            token = next(
                (
                    operator
                    for operator in (
                        "===", "!==", "=>", "?.", "...", "&&", "||", "??", "++", "--"
                    )
                    if source.startswith(operator, index)
                ),
                character,
            )
            index += len(token)
            kind = "punctuation"
        previous_token = token
        yield token, kind, start, index


class _InlineScriptCollector(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = array("I", [0])
        self.line_starts.extend(
            index + 1 for index, character in enumerate(source) if character == "\n"
        )
        self.scripts: list[tuple[int, int, int, int, dict[str, str | None]]] = []
        self.open_script: tuple[int, int, dict[str, str | None]] | None = None

    def _source_offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "script" and self.open_script is None:
            start = self._source_offset()
            start_tag = self.get_starttag_text() or ""
            self.open_script = (start, start + len(start_tag), dict(attrs))

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "script":
            super().handle_startendtag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag != "script" or self.open_script is None:
            return
        closing_start = self._source_offset()
        closing_end = self.source.find(">", closing_start)
        if closing_end >= 0:
            start, content_start, attrs = self.open_script
            self.scripts.append(
                (start, content_start, closing_start, closing_end + 1, attrs)
            )
        self.open_script = None

router = APIRouter(prefix=DEBUG_PREFIX, include_in_schema=False)


class NoStoreStaticFiles(StaticFiles):
    """Prevent browsers from reusing stale debug UI scripts and styles."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


def _debug_asset_version() -> str:
    """Return a content hash for the debug UI styles and script."""
    digest = sha256()
    for asset_name in ("style.css", "app.js"):
        digest.update((STATIC_DIR / asset_name).read_bytes())
    return digest.hexdigest()[:12]


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


def _principal_needs_refresh(principal: Principal) -> bool:
    """Refresh only from expiry obtained from an already verified JWT."""
    return (
        principal.expires_at is not None
        and principal.expires_at <= time() + REFRESH_LEEWAY_SECONDS
    )


def _debug_client_auth() -> httpx.BasicAuth:
    secret = settings.debug_ui_keycloak_client_secret
    if secret is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cliente de debug do Keycloak não configurado.",
        )
    return httpx.BasicAuth(
        settings.debug_ui_keycloak_client_id,
        secret.get_secret_value(),
    )


def _oauth_error(response: httpx.Response) -> str | None:
    try:
        document = response.json()
    except ValueError:
        return None
    error = document.get("error") if isinstance(document, dict) else None
    return error if isinstance(error, str) else None


async def _refresh_debug_session(
    refresh_token: str,
    response: Response,
) -> Principal:
    try:
        async with httpx.AsyncClient(
            timeout=settings.debug_ui_request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            upstream = await client.post(
                settings.effective_debug_ui_token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                auth=_debug_client_auth(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keycloak indisponível.",
        ) from exc

    if (
        upstream.status_code in {400, 401}
        and _oauth_error(upstream) == "invalid_grant"
    ):
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
            detail="Keycloak devolveu uma resposta de refresh inválida.",
        ) from exc

    try:
        principal = await principal_from_token(access_token)
    except HTTPException as exc:
        if exc.status_code != status.HTTP_401_UNAUTHORIZED:
            raise
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak devolveu um token renovado incompatível.",
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
            if not _principal_needs_refresh(principal):
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
async def debug_index(request: Request) -> Response:
    _require_enabled()
    if request.query_params:
        return RedirectResponse(
            url=request.url.path,
            status_code=status.HTTP_303_SEE_OTHER,
            headers={
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
            },
        )

    version = _debug_asset_version()
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace(
        'href="/debug/assets/style.css"',
        f'href="/debug/assets/style.css?v={version}"',
    )
    html = html.replace(
        'src="/debug/assets/app.js"',
        f'src="/debug/assets/app.js?v={version}"',
    )
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/api/login")
async def debug_login(
    payload: DebugLoginRequest,
    response: Response,
) -> DebugSessionResponse:
    _require_enabled()
    try:
        async with httpx.AsyncClient(
            timeout=settings.debug_ui_request_timeout_seconds,
            follow_redirects=False,
        ) as client:
            upstream = await client.post(
                settings.effective_debug_ui_token_url,
                data={
                    "grant_type": "password",
                    "username": payload.email,
                    "password": payload.password.get_secret_value(),
                    "scope": "openid ouros-identity",
                },
                auth=_debug_client_auth(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keycloak indisponível.",
        ) from exc

    if (
        upstream.status_code in {400, 401}
        and _oauth_error(upstream) == "invalid_grant"
    ):
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
            detail="Keycloak devolveu uma resposta inválida.",
        ) from exc

    try:
        principal = await principal_from_token(access_token)
    except HTTPException as exc:
        if exc.status_code != status.HTTP_401_UNAUTHORIZED:
            raise
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak devolveu um token incompatível com o AI Server.",
        ) from exc

    _set_access_cookie(response, access_token, expires_in)
    _set_refresh_cookie(response, refresh_token, refresh_expires_in)
    return _session(principal)


@router.post("/api/logout", status_code=status.HTTP_204_NO_CONTENT)
async def debug_logout(response: Response) -> None:
    _require_enabled()
    _delete_session_cookies(response)


@router.get("/api/conversations")
async def debug_conversations(
    request: Request,
    principal: Annotated[Principal, Depends(_debug_principal)],
) -> DebugConversationListResponse:
    owner_store = getattr(request.app.state, "thread_ownership", None)
    records = (
        await owner_store.list_for_user(principal.user_id, limit=40)
        if owner_store is not None
        else []
    )
    return DebugConversationListResponse(
        conversations=[
            {
                "id": record["thread_id"],
                "title": record.get("title") or f"Conversa {record['thread_id'][:8]}",
            }
            for record in records
            if isinstance(record.get("thread_id"), str)
        ]
    )


@router.get("/api/conversations/{thread_id}")
async def debug_conversation_history(
    thread_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(_debug_principal)],
) -> HistoryResponse:
    return await get_thread_history(
        request.app.state.checkpointer,
        thread_id,
        principal.user_id,
        limit=100,
        before=None,
        visualization_store=getattr(request.app.state, "visualization_store", None),
    )


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
        visualization_store=getattr(
            request.app.state,
            "visualization_store",
            None,
        ),
    )
    return DebugChatResponse(
        thread_id=response.thread_id,
        message=response.message,
        agents=response.agents,
        tools=response.tools,
        visualizations=response.visualizations,
        **diagnostics,
    )


@router.post("/api/visualization")
async def debug_visualization(
    request: Request,
    _principal: Annotated[Principal, Depends(_debug_principal)],
) -> HTMLResponse:
    """Render a chart in an isolated iframe with a Plotly-compatible CSP."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type.strip().lower() != "application/x-www-form-urlencoded":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_DEBUG_VISUALIZATION_FORM_BYTES:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
    try:
        form = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=1,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formulário de gráfico inválido.",
        ) from error
    html_values = form.get("html", [])
    html = html_values[0] if len(html_values) == 1 else None
    if not isinstance(html, str) or not html or len(html) > MAX_DEBUG_VISUALIZATION_HTML_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="HTML de gráfico inválido.",
        )

    html = _wrap_inline_chart_scripts(html)
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; "
                "script-src 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly; "
                "style-src 'unsafe-inline'; img-src data: blob:; "
                "font-src data:; connect-src 'none'; object-src 'none'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


def _has_top_level_bare_return(script: str) -> bool:
    """Detect top-level ``return;`` without rescanning function-like text."""
    previous_token = ""
    before_previous_token = ""
    previous_kind = ""
    function_parameters_pending = False
    function_body_pending = False
    arrow_body_pending = False
    function_parameters = bytearray()
    function_bodies = bytearray()
    class_bodies = bytearray()
    class_body_pending: list[tuple[int, int]] = []
    bracket_depth = 0
    function_depth = 0
    bare_return_end: int | None = None

    for token, kind, start, end in _iter_js_tokens(script):
        if bare_return_end is not None:
            between = script[bare_return_end:start]
            if token == ";" and function_depth == 0 and all(
                character.isspace() for character in between
            ):
                return True
            bare_return_end = None

        if token != "{":
            function_body_pending = False
            arrow_body_pending = False

        if token == "(":
            is_method = (
                previous_kind == "identifier"
                and previous_token not in _CONTROL_PAREN_TOKENS
                and (
                    before_previous_token in _METHOD_BOUNDARY_TOKENS
                    or (
                        before_previous_token == ";"
                        and bool(class_bodies and class_bodies[-1])
                    )
                )
            )
            function_parameters.append(function_parameters_pending or is_method)
            function_parameters_pending = False
        elif token == ")":
            if function_parameters and function_parameters.pop():
                function_body_pending = True
        elif token == "[":
            bracket_depth += 1
        elif token == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif token == "=>":
            arrow_body_pending = True
        elif token == "{":
            class_header = (len(function_parameters), bracket_depth)
            is_class_body = (
                bool(class_body_pending)
                and class_body_pending[-1] == class_header
            )
            if is_class_body:
                class_body_pending.pop()
            class_bodies.append(is_class_body)

            is_function_body = function_body_pending or arrow_body_pending
            function_bodies.append(is_function_body)
            if is_function_body:
                function_depth += 1
            function_body_pending = False
            arrow_body_pending = False
        elif token == "}":
            if class_bodies:
                class_bodies.pop()
            if function_bodies and function_bodies.pop():
                function_depth -= 1

        if token == "class" and previous_token not in {".", "?."}:
            class_body_pending.append((len(function_parameters), bracket_depth))
        elif (
            class_body_pending
            and class_body_pending[-1] == (len(function_parameters), bracket_depth)
            and token in {";", ":", "="}
        ):
            class_body_pending.pop()

        if token == "function":
            function_parameters_pending = True
        elif function_parameters_pending and token in {";", "="}:
            function_parameters_pending = False

        if (
            token == "return"
            and function_depth == 0
            and previous_token not in {".", "?."}
        ):
            bare_return_end = end

        before_previous_token, previous_token = previous_token, token
        previous_kind = kind

    return False


def _wrap_inline_chart_scripts(html: str) -> str:
    """Wrap inline JavaScript scripts that contain a top-level bare return."""
    parser = _InlineScriptCollector(html)
    parser.feed(html)
    parser.close()

    parts: list[str] = []
    cursor = 0
    for _, content_start, content_end, _, attrs in parser.scripts:
        script = html[content_start:content_end]
        script_type = (attrs.get("type") or "").strip("\t\n\f\r ").lower()
        if (
            "src" in attrs
            or (script_type and script_type not in _SCRIPT_TYPES)
            or not _has_top_level_bare_return(script)
        ):
            continue
        parts.extend(
            (
                html[cursor:content_start],
                "(()=>{\n",
                script,
                "\n})();",
            )
        )
        cursor = content_end
    if not parts:
        return html
    parts.append(html[cursor:])
    return "".join(parts)


def install_debug_ui(app: FastAPI) -> None:
    """Mount the optional debug product without coupling it to normal routes."""

    if not settings.debug_ui_enabled:
        return
    app.include_router(router)
    app.mount(
        "/debug/assets",
        NoStoreStaticFiles(directory=STATIC_DIR),
        name="debug-ui-assets",
    )
