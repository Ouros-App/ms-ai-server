import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.services.history import get_thread_history


class HistoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_restores_visualizations_separately_from_checkpoint(self) -> None:
        messages = [
            HumanMessage(content="Gere um gráfico"),
            AIMessage(
                content="Pronto, gerei o gráfico.",
                additional_kwargs={"visualization_id": "viz-1"},
            ),
        ]
        checkpoint = SimpleNamespace(
            checkpoint={
                "channel_values": {
                    "user_id": "user-1",
                    "messages": messages,
                }
            }
        )
        checkpointer = SimpleNamespace(aget_tuple=AsyncMock(return_value=checkpoint))
        visualization_store = SimpleNamespace(
            get_many=AsyncMock(
                return_value={
                    "viz-1": [
                        {
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
                    ]
                }
            )
        )

        history = await get_thread_history(
            checkpointer,
            "thread-1",
            "user-1",
            limit=20,
            before=None,
            visualization_store=visualization_store,
        )

        self.assertEqual(history.messages[0].visualizations, [])
        self.assertEqual(history.messages[1].visualizations[0].title, "Consumo mensal")
        self.assertEqual(
            history.messages[1].visualizations[0].charts[0].html,
            "<html>chart</html>",
        )
        visualization_store.get_many.assert_awaited_once_with(
            "thread-1", "user-1", ["viz-1"]
        )

    async def test_missing_thread_does_not_load_visualizations_for_user(self) -> None:
        checkpointer = SimpleNamespace(aget_tuple=AsyncMock(return_value=None))

        with self.assertRaises(HTTPException) as error:
            await get_thread_history(
                checkpointer,
                "thread-1",
                "user-1",
                limit=20,
                before=None,
            )

        self.assertEqual(error.exception.status_code, 404)
