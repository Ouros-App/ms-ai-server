import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.agents.model import get_chat_model
from app.api.routes import router
from app.core.auth import Principal, get_current_principal
from app.core.config import settings


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        for attribute, value in (
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

        self.principal = Principal(
            subject="keycloak-subject-42",
            user_id="42",
            user_type="farm_owner",
            access_token="signed-keycloak-token",
        )

        async def principal_override() -> Principal:
            return self.principal

        self.principal_override = principal_override
        app.dependency_overrides[get_current_principal] = principal_override
        self.client = TestClient(app)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer signed-keycloak-token"}

    def test_chat_and_history_derive_identity_from_jwt(self) -> None:
        first = self.client.post(
            "/v1/chat",
            json={"thread_id": "thread", "message": "Como funciona o ranking?"},
            headers=self.auth_headers(),
        )
        second = self.client.post(
            "/v1/chat",
            json={"thread_id": "thread", "message": "E depois?"},
            headers=self.auth_headers(),
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["thread_id"], "thread")
        self.assertEqual(first.json()["agents"], ["router", "ranking", "default"])
        self.assertEqual(second.status_code, 200)

        history = self.client.get(
            "/v1/chat/thread/history",
            params={"limit": 2},
            headers=self.auth_headers(),
        )
        self.assertEqual(history.status_code, 200)
        history_body = history.json()
        self.assertEqual(
            [item["role"] for item in history_body["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(history_body["next_cursor"], "2")

        previous_page = self.client.get(
            "/v1/chat/thread/history",
            params={"limit": 2, "before": history_body["next_cursor"]},
            headers=self.auth_headers(),
        )
        self.assertEqual(previous_page.status_code, 200)
        self.assertIsNone(previous_page.json()["next_cursor"])

    def test_compatibility_user_id_must_match_signed_database_id(self) -> None:
        matching = self.client.post(
            "/v1/chat",
            json={
                "user_id": "42",
                "thread_id": "matching",
                "message": "ranking",
            },
            headers=self.auth_headers(),
        )
        mismatched = self.client.post(
            "/v1/chat",
            json={
                "user_id": "99",
                "thread_id": "mismatch",
                "message": "ranking",
            },
            headers=self.auth_headers(),
        )

        self.assertEqual(matching.status_code, 200)
        self.assertEqual(mismatched.status_code, 403)
        self.assertEqual(
            mismatched.json()["detail"],
            "O user_id nao corresponde ao usuario autenticado.",
        )

    def test_history_compatibility_user_id_must_match_claim(self) -> None:
        response = self.client.get(
            "/v1/chat/unknown/history",
            params={"user_id": "99"},
            headers=self.auth_headers(),
        )

        self.assertEqual(response.status_code, 403)

    def test_thread_cannot_be_reused_by_another_signed_identity(self) -> None:
        first = self.client.post(
            "/v1/chat",
            json={"thread_id": "owned-thread", "message": "ranking"},
            headers=self.auth_headers(),
        )
        self.assertEqual(first.status_code, 200)

        self.principal = Principal(
            subject="keycloak-subject-43",
            user_id="43",
            user_type="farm_owner",
            access_token="signed-keycloak-token-43",
        )
        second = self.client.post(
            "/v1/chat",
            json={"thread_id": "owned-thread", "message": "ranking"},
            headers=self.auth_headers(),
        )

        self.assertEqual(second.status_code, 403)

    def test_unknown_thread_history_is_not_found(self) -> None:
        response = self.client.get(
            "/v1/chat/unknown/history",
            headers=self.auth_headers(),
        )
        self.assertEqual(response.status_code, 404)

    def test_root_health_and_metrics_work_for_authenticated_principal(self) -> None:
        self.assertEqual(
            self.client.get("/", headers=self.auth_headers()).json(),
            {"message": "AI Server is running"},
        )
        self.assertEqual(
            self.client.get("/health", headers=self.auth_headers()).json(),
            {"status": "ok"},
        )
        metrics = self.client.get("/metrics", headers=self.auth_headers())
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("ai_server_http_requests_total", metrics.text)

    def test_missing_bearer_is_rejected_by_real_auth_dependency(self) -> None:
        self.app.dependency_overrides.pop(get_current_principal)
        try:
            response = self.client.get("/")
        finally:
            self.app.dependency_overrides[get_current_principal] = self.principal_override

        self.assertEqual(response.status_code, 401)

    def test_invalid_bearer_has_no_legacy_fallback(self) -> None:
        self.app.dependency_overrides.pop(get_current_principal)
        try:
            with patch("app.core.auth._decode_keycloak_token", return_value=None):
                response = self.client.post(
                    "/v1/chat",
                    json={"thread_id": "invalid", "message": "ranking"},
                    headers={"Authorization": "Bearer old-shared-token"},
                )
        finally:
            self.app.dependency_overrides[get_current_principal] = self.principal_override

        self.assertEqual(response.status_code, 401)

    def test_route_forwards_validated_access_token_to_graph(self) -> None:
        with patch(
            "app.api.routes.invoke_graph",
            new=AsyncMock(),
        ) as invoke:
            # Return type validation expects a ChatResponse, so use the actual model.
            from app.schemas.chat import ChatResponse

            invoke.return_value = ChatResponse(
                thread_id="forward",
                message="ok",
                agents=["default"],
                tools=[],
            )
            response = self.client.post(
                "/v1/chat",
                json={"thread_id": "forward", "message": "teste"},
                headers=self.auth_headers(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(invoke.call_args.kwargs["principal_token"], "signed-keycloak-token")
        self.assertEqual(invoke.call_args.args[2], "42")

    def test_chat_returns_controlled_503_when_graph_times_out(self) -> None:
        graph = Mock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
        graph.ainvoke = AsyncMock(side_effect=TimeoutError())
        self.app.state.graph = graph

        with patch.object(settings, "llm_total_timeout_seconds", 20):
            response = self.client.post(
                "/v1/chat",
                json={"thread_id": "timeout", "message": "ranking"},
                headers=self.auth_headers(),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "O provedor de IA demorou para responder. Tente novamente.",
        )


if __name__ == "__main__":
    unittest.main()
