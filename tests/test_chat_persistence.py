import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from app.schemas.chat import ChatRequest
from app.services.chat import invoke_graph


class ChatPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_message_is_not_saved_as_conversation_title(self) -> None:
        graph = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(values={})),
            ainvoke=AsyncMock(),
        )
        thread_ownership = SimpleNamespace(
            claim=AsyncMock(return_value=True),
            set_title_if_missing=AsyncMock(),
        )
        guardrail = SimpleNamespace(
            allowed=False,
            category="unsupported",
            message="Não posso ajudar com isso.",
            as_state=lambda: {"allowed": False},
        )

        with patch(
            "app.services.chat.guard_input",
            new=AsyncMock(return_value=guardrail),
        ):
            response = await invoke_graph(
                graph,
                ChatRequest(
                    user_id="user-1",
                    thread_id="thread-1",
                    message="mensagem com dados privados",
                ),
                "user-1",
                thread_ownership=thread_ownership,
            )

        self.assertEqual(response.message, guardrail.message)
        thread_ownership.claim.assert_awaited_once_with("thread-1", "user-1")
        thread_ownership.set_title_if_missing.assert_not_awaited()
        graph.ainvoke.assert_not_awaited()

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
        thread_ownership = SimpleNamespace(
            claim=AsyncMock(return_value=True),
            set_title_if_missing=AsyncMock(return_value=True),
        )
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
                thread_ownership=thread_ownership,
                visualization_store=visualization_store,
            )

        self.assertEqual(response.visualizations[0].title, "Consumo mensal")
        thread_ownership.claim.assert_awaited_once_with("thread-1", "user-1")
        thread_ownership.set_title_if_missing.assert_awaited_once_with(
            "thread-1", "user-1", "Gere um gráfico"
        )
        visualization_store.save.assert_awaited_once_with(
            "thread-1", "user-1", "viz-1", [visualization]
        )

    async def test_import_context_is_hidden_from_user_message_and_replaced(self) -> None:
        previous_context = SystemMessage(
            id="import-context-1",
            content='{"records":[{"value":"old"}]}',
            additional_kwargs={"measurement_import_context": True},
        )
        graph = SimpleNamespace(
            aget_state=AsyncMock(
                return_value=SimpleNamespace(values={"messages": [previous_context]})
            ),
            ainvoke=AsyncMock(
                return_value={
                    "messages": [AIMessage(content="Revise antes de confirmar.")],
                    "agents": ["default"],
                    "tools": [],
                }
            ),
        )
        thread_ownership = SimpleNamespace(
            claim=AsyncMock(return_value=True),
            set_title_if_missing=AsyncMock(),
        )
        guardrail = SimpleNamespace(
            allowed=True,
            category="allowed",
            sanitized_text="Importe meu histórico",
            as_state=lambda: {"allowed": True},
        )

        with patch(
            "app.services.chat.guard_input",
            new=AsyncMock(return_value=guardrail),
        ):
            await invoke_graph(
                graph,
                ChatRequest(
                    user_id="user-1",
                    thread_id="thread-1",
                    message="Importe meu histórico",
                ),
                "user-1",
                thread_ownership=thread_ownership,
                system_context='{"records":[{"value":"new"}]}',
            )

        graph_messages = graph.ainvoke.call_args.args[0]["messages"]
        self.assertIsInstance(graph_messages[0], RemoveMessage)
        self.assertEqual(graph_messages[0].id, "import-context-1")
        self.assertIsInstance(graph_messages[1], SystemMessage)
        self.assertIn('"value":"new"', graph_messages[1].content)
        self.assertIsInstance(graph_messages[2], HumanMessage)
        self.assertEqual(graph_messages[2].content, "Importe meu histórico")
