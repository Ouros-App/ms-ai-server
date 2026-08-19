import unittest

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
        self.previous_token = settings.auth_bearer_token
        self.previous_groq_key = settings.groq_api_key
        self.previous_nvidia_key = settings.nvidia_api_key
        settings.auth_bearer_token = SecretStr("test-token")
        settings.groq_api_key = None
        settings.nvidia_api_key = None
        get_chat_model.cache_clear()
        app = FastAPI()
        app.state.graph = build_graph(InMemorySaver())
        app.include_router(router)
        self.app = app
        self.client = TestClient(app)

    def tearDown(self) -> None:
        settings.auth_bearer_token = self.previous_token
        settings.groq_api_key = self.previous_groq_key
        settings.nvidia_api_key = self.previous_nvidia_key
        get_chat_model.cache_clear()

    def token(self) -> str:
        return "test-token"

    def test_chat_and_health(self) -> None:
        self.assertEqual(self.client.get("/").json(), {"message": "AI Server is running"})
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

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

    def test_chat_requires_user_id(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"thread_id": "thread", "message": "teste"},
            headers={"Authorization": f"Bearer {self.token()}"},
        )

        self.assertEqual(response.status_code, 422)

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

    def test_chat_requires_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
        )

        self.assertEqual(response.status_code, 401)

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
