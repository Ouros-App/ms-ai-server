from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.auth import Principal
from app.core.config import settings
from app.debug_ui.router import COOKIE_NAME, install_debug_ui
from app.debug_ui.trace import capture_debug_trace, trace_event
from app.schemas.chat import ChatResponse


def build_debug_app() -> FastAPI:
    app = FastAPI()
    app.state.graph = object()
    app.state.thread_ownership = object()
    install_debug_ui(app)
    return app


def test_debug_ui_is_not_mounted_when_disabled() -> None:
    app = FastAPI()
    with patch.object(settings, "debug_ui_enabled", False):
        install_debug_ui(app)

    assert all(getattr(route, "path", "") != "/debug" for route in app.routes)


def test_debug_ui_serves_console_when_enabled() -> None:
    with patch.object(settings, "debug_ui_enabled", True):
        app = build_debug_app()
        with TestClient(app) as client:
            response = client.get("/debug")
            styles = client.get("/debug/assets/style.css")

    assert response.status_code == 200
    assert "Debug Console" in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert styles.status_code == 200
    assert "grid-template-rows: 68px minmax(0, 1fr) auto auto" in styles.text
    assert ".messages { min-height: 0;" in styles.text


def test_debug_session_requires_cookie() -> None:
    with patch.object(settings, "debug_ui_enabled", True):
        app = build_debug_app()
        with TestClient(app) as client:
            response = client.get("/debug/api/session")

    assert response.status_code == 401


def test_login_proxies_credentials_and_sets_http_only_cookie() -> None:
    principal = Principal(
        subject="keycloak-user",
        user_id="42",
        user_type="farm_owner",
        access_token="signed-access-token",
    )
    captured = {}
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "access_token": "signed-access-token",
                "expires_in": 600,
                "refresh_token": "refresh-never-forwarded",
                "token_type": "Bearer",
            },
        )

    def client_factory(**kwargs):
        return real_async_client(
            transport=httpx.MockTransport(handler),
            timeout=kwargs.get("timeout"),
            follow_redirects=kwargs.get("follow_redirects", False),
        )

    with (
        patch.object(settings, "debug_ui_enabled", True),
        patch.object(settings, "debug_ui_cookie_secure", False),
        patch("app.debug_ui.router.httpx.AsyncClient", side_effect=client_factory),
        patch(
            "app.debug_ui.router.principal_from_token",
            new=AsyncMock(return_value=principal),
        ),
    ):
        app = build_debug_app()
        with TestClient(app) as client:
            response = client.post(
                "/debug/api/login",
                json={"email": "User@Example.com", "password": "Senha123!"},
            )
            session = client.get("/debug/api/session")

    assert response.status_code == 200
    assert response.json() == {
        "authenticated": True,
        "user_id": "42",
        "account_type": "farm_owner",
    }
    assert "signed-access-token" not in response.text
    assert "refresh-never-forwarded" not in response.text
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert captured["path"].endswith("/v1/auth/token")
    assert "user%40example.com" not in captured["body"]
    assert '"email":"user@example.com"' in captured["body"]
    assert session.status_code == 200


def test_debug_chat_uses_authenticated_identity_and_returns_trace() -> None:
    principal = Principal(
        subject="keycloak-user",
        user_id="42",
        user_type="admin",
        access_token="signed-access-token",
    )
    chat_response = ChatResponse(
        thread_id="thread-1",
        message="**resultado**",
        agents=["router", "ranking", "default"],
        tools=["get_user_context"],
    )
    diagnostics = {
        "routes": ["ranking"],
        "specialist_results": [{"agent": "ranking", "status": "ok"}],
        "guardrail": {"allowed": True},
        "trace": [{"t_ms": 1.2, "event": "router.selected"}],
        "duration_ms": 123.4,
    }

    with (
        patch.object(settings, "debug_ui_enabled", True),
        patch.object(settings, "debug_ui_cookie_secure", False),
        patch(
            "app.debug_ui.router.principal_from_token",
            new=AsyncMock(return_value=principal),
        ),
        patch(
            "app.debug_ui.router.invoke_graph_debug",
            new=AsyncMock(return_value=(chat_response, diagnostics)),
        ) as invoke,
    ):
        app = build_debug_app()
        with TestClient(app) as client:
            client.cookies.set(COOKIE_NAME, "signed-access-token", path="/debug")
            response = client.post(
                "/debug/api/chat",
                json={"message": "teste", "thread_id": "thread-1"},
            )

    assert response.status_code == 200
    assert response.json()["specialist_results"][0]["agent"] == "ranking"
    assert response.json()["trace"][0]["event"] == "router.selected"
    args = invoke.await_args.args
    assert args[1].user_id == "42"
    assert args[2] == "42"
    assert invoke.await_args.kwargs["principal_token"] == "signed-access-token"


def test_trace_capture_is_request_local_and_sanitized() -> None:
    trace_event("ignored", token="secret")
    with capture_debug_trace() as events:
        trace_event(
            "agent.response",
            response="ok",
            values=[1, 2, 3],
            authorization="Bearer very-secret",
            result={
                "safe": "visible",
                "access_token": "hidden",
                "nested": {"password": "hidden-too"},
            },
        )

    assert len(events) == 1
    assert events[0]["event"] == "agent.response"
    assert events[0]["response"] == "ok"
    assert events[0]["values"] == [1, 2, 3]
    assert events[0]["authorization"] == "[REDACTED]"
    assert events[0]["result"]["safe"] == "visible"
    assert events[0]["result"]["access_token"] == "[REDACTED]"
    assert events[0]["result"]["nested"]["password"] == "[REDACTED]"
