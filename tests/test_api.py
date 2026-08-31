import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import jwt
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from app.agents.graph import build_graph
from app.agents.model import get_chat_model
from app.api.routes import router
from app.core.config import settings


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        for attribute, value in (
            ("auth_bearer_token", SecretStr("test-token")),
            ("auth_jwt_secret", None),
            ("auth_require_user_jwt", False),
            ("groq_api_key", None),
            ("nvidia_api_key", None),
        ):
            patcher = patch.object(settings, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        get_chat_model.cache_clear()
        self.addCleanup(get_chat_model.cache_clear)
        app = FastAPI()
        self.checkpointer = InMemorySaver()
        app.state.checkpointer = self.checkpointer
        app.state.thread_ownership = None
        app.state.graph = build_graph(self.checkpointer)
        app.include_router(router)
        self.app = app
        self.client = TestClient(app)

    def token(self) -> str:
        return "test-token"

    def test_chat_and_health(self) -> None:
        headers = {"Authorization": f"Bearer {self.token()}"}
        self.assertEqual(self.client.get("/", headers=headers).json(), {"message": "AI Server is running"})
        self.assertEqual(self.client.get("/health", headers=headers).json(), {"status": "ok"})

        first = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "Como funciona o ranking?"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        second = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "E depois?"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(first.status_code, 200)
        body = first.json()
        self.assertEqual(body["thread_id"], "thread")
        self.assertIsInstance(body["message"], str)
        self.assertEqual(body["agents"], ["router", "default"])
        self.assertEqual(body["tools"], [])
        self.assertEqual(second.status_code, 200)

        history = self.client.get(
            "/v1/chat/thread/history",
            params={"user_id": "user-1", "limit": 2},
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        self.assertEqual(history.status_code, 200)
        history_body = history.json()
        self.assertEqual([item["role"] for item in history_body["messages"]], ["user", "assistant"])
        self.assertEqual(history_body["next_cursor"], "2")

        previous_page = self.client.get(
            "/v1/chat/thread/history",
            params={"user_id": "user-1", "limit": 2, "before": history_body["next_cursor"]},
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        self.assertEqual(previous_page.status_code, 200)
        self.assertIsNone(previous_page.json()["next_cursor"])

    def test_root_and_health_require_bearer_token(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 401)

        headers = {"Authorization": f"Bearer {self.token()}"}
        self.assertEqual(self.client.get("/", headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/health", headers=headers).status_code, 200)

    def test_chat_requires_user_id(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"thread_id": "thread", "message": "teste"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_shared_token_when_user_jwt_is_required(self) -> None:
        with patch.object(settings, "auth_require_user_jwt", True):
            response = self.client.post(
                "/v1/chat",
                json={"user_id": "6", "thread_id": "thread", "message": "ranking"},
                headers={"Authorization": f"Bearer {self.token()}"},
            )

        self.assertEqual(response.status_code, 403)

    def test_history_rejects_shared_token_when_user_jwt_is_required(self) -> None:
        with patch.object(settings, "auth_require_user_jwt", True):
            response = self.client.get(
                "/v1/chat/thread/history",
                params={"user_id": "6"},
                headers={"Authorization": f"Bearer {self.token()}"},
            )

        self.assertEqual(response.status_code, 403)

    def test_chat_uses_user_id_from_jwt_and_rejects_mismatch(self) -> None:
        secret = "j" * 32
        token = jwt.encode(
            {"sub": "6", "user_type": "farm_owner"},
            secret,
            algorithm="HS256",
        )
        with (
            patch.object(settings, "auth_jwt_secret", SecretStr(secret)),
            patch.object(settings, "auth_require_user_jwt", True),
        ):
            matching = self.client.post(
                "/v1/chat",
                json={"user_id": "6", "thread_id": "jwt-thread", "message": "ranking"},
                headers={"Authorization": f"Bearer {token}"},
            )
            mismatched = self.client.post(
                "/v1/chat",
                json={"user_id": "7", "thread_id": "other-thread", "message": "ranking"},
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(matching.status_code, 200)
        self.assertEqual(mismatched.status_code, 403)
        self.assertEqual(
            mismatched.json()["detail"],
            "O user_id nao corresponde ao usuario autenticado.",
        )

    def test_chat_preserves_distinct_jwt_subject_and_user_id(self) -> None:
        secret = "j" * 32
        token = jwt.encode(
            {"sub": "subject-6", "user_id": "6", "user_type": "farm_owner"},
            secret,
            algorithm="HS256",
        )
        with (
            patch.object(settings, "auth_jwt_secret", SecretStr(secret)),
            patch.object(settings, "auth_require_user_jwt", True),
        ):
            response = self.client.post(
                "/v1/chat",
                json={"user_id": "6", "thread_id": "jwt-user-id", "message": "ranking"},
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(response.status_code, 200)

    def test_chat_rejects_thread_for_another_user(self) -> None:
        first = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "Como funciona o ranking?"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        second = self.client.post(
            "/v1/chat",
            json={"user_id": "user-2", "thread_id": "thread", "message": "Como funciona o ranking?"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 403)

        history = self.client.get(
            "/v1/chat/thread/history",
            params={"user_id": "user-2"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        self.assertEqual(history.status_code, 403)

    def test_history_returns_not_found_for_unknown_thread(self) -> None:
        response = self.client.get(
            "/v1/chat/unknown/history",
            params={"user_id": "user-1"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(response.status_code, 404)

    def test_chat_requires_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
        )

        self.assertEqual(response.status_code, 401)

    def test_metrics_requires_bearer_and_returns_prometheus(self) -> None:
        unauthenticated = self.client.get("/metrics")
        self.assertEqual(unauthenticated.status_code, 401)

        response = self.client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("ai_server_http_requests_total", response.text)

    def test_chat_rejects_invalid_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
            headers={"Authorization": "Bearer invalid"},
        )

        self.assertEqual(response.status_code, 401)

    def test_chat_rejects_wrong_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
            headers={"Authorization": "Bearer wrong-token"},
        )

        self.assertEqual(response.status_code, 401)

    def test_chat_rejects_when_authentication_is_not_configured(self) -> None:
        with patch.object(settings, "auth_bearer_token", None):
            response = self.client.post(
                "/v1/chat",
                json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
                headers={"Authorization": "Bearer test-token"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Autenticacao nao configurada.")

    def test_chat_returns_controlled_503_when_graph_times_out(self) -> None:
        graph = Mock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
        graph.ainvoke = AsyncMock(side_effect=TimeoutError())
        self.app.state.graph = graph

        with patch.object(settings, "llm_total_timeout_seconds", 20):
            response = self.client.post(
                "/v1/chat",
                json={"user_id": "user-1", "thread_id": "timeout", "message": "ranking"},
                headers={"Authorization": f"Bearer {self.token()}"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "O provedor de IA demorou para responder. Tente novamente.",
        )
