import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.api.routes import router


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.state.graph = build_graph(InMemorySaver())
        app.include_router(router)
        self.client = TestClient(app)

    def test_chat_and_health(self) -> None:
        self.assertEqual(self.client.get("/").json(), {"message": "AI Server is running"})
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

        first = self.client.post(
            "/v1/chat",
            json={"user_id": "user-1", "thread_id": "thread", "message": "teste"},
        )
        second = self.client.post(
            "/v1/chat",
            json={"user_id": "user-2", "thread_id": "thread", "message": "teste"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["agents"], ["router", "default"])
        self.assertEqual(second.status_code, 403)
