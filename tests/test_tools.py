import unittest

from app.agents.tools import build_memory_tools


class FakeMemoryStore:
    def __init__(self) -> None:
        self.memories: dict[str, list[str]] = {}

    async def list(self, user_id: str, limit: int = 20) -> list[str]:
        return self.memories.get(user_id, [])[:limit]

    async def save(self, user_id: str, memory: str) -> None:
        self.memories.setdefault(user_id, []).append(memory)


class MemoryToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_read_and_save_for_injected_user(self) -> None:
        store = FakeMemoryStore()
        tools = build_memory_tools(store, "user-1")
        save_tool, recall_tool = tools[1], tools[0]

        await save_tool.ainvoke({"memory": "Prefere respostas curtas."})
        memories = await recall_tool.ainvoke({})

        self.assertEqual(memories, "- Prefere respostas curtas.")
        self.assertEqual(store.memories, {"user-1": ["Prefere respostas curtas."]})

    async def test_tool_rejects_credential_memory(self) -> None:
        store = FakeMemoryStore()
        save_tool = build_memory_tools(store, "user-1")[1]

        result = await save_tool.ainvoke({"memory": "token: segredo-123"})

        self.assertIn("Memoria recusada", result)
        self.assertEqual(store.memories, {})
