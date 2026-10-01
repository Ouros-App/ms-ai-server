import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph


class ChatPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_created_visualization_is_saved_by_message_reference(self) -> None:
        visualization = {
            "type": "ouros_dashboard",
            "title": "Consumo mensal",
            "charts": [
                {
                    "id": "chart-1",
                    "title": "Consumo mensal",
                    "render_as": "bar",
                    "html": "<html>chart</html>",
                }
            ],
        }
        graph = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values={})),
            ainvoke=AsyncMock(
                return_value={
                    "messages": [
                        AIMessage(
                            content="Pronto, gerei o gráfico.",
                            additional_kwargs={"visualization_id": "viz-1"},
                        )
                    ],
                    "agents": ["router", "visualization", "default"],
                    "tools": ["create_custom_dashboard"],
                }
            ),
        )
        visualization_store = SimpleNamespace(save=AsyncMock())
        guardrail = SimpleNamespace(
            allowed=True,
            category="allowed",
            sanitized_text="Gere um gráfico",
            as_state=lambda: {"allowed": True},
        )

        @contextmanager
        def captured_visualizations():
            yield [visualization]

        with (
            patch("app.services.chat.guard_input", new=AsyncMock(return_value=guardrail)),
            patch(
                "app.services.chat.capture_mcp_visualizations",
                side_effect=captured_visualizations,
            ),
        ):
            response = await invoke_graph(
                graph,
                ChatRequest(user_id="user-1", thread_id="thread-1", message="Gere um gráfico"),
                "user-1",
                visualization_store=visualization_store,
            )

        self.assertEqual(response.visualizations[0].title, "Consumo mensal")
        visualization_store.save.assert_awaited_once_with(
            "thread-1", "user-1", "viz-1", [visualization]
        )
