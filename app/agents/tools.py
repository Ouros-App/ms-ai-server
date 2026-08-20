import logging

from langchain_core.tools import tool

from app.agents.guardrails import memory_is_allowed
from app.repositories.memory import UserMemoryStore

logger = logging.getLogger(__name__)


def build_memory_tools(memory_store: UserMemoryStore, user_id: str) -> list:
    """Cria tools com o user_id injetado pelo backend, fora do controle do modelo."""

    @tool
    async def recall_user_memories() -> str:
        """Recupera memorias persistentes do usuario em outras conversas."""
        try:
            memories = await memory_store.list(user_id)
        except Exception:
            logger.exception("memory_tool_failed tool=recall_user_memories")
            return "Nao foi possivel consultar as memorias agora."
        logger.info("memory_tool_called tool=recall_user_memories count=%d", len(memories))
        if not memories:
            return "Nenhuma memoria persistente encontrada para este usuario."
        return "\n".join(f"- {memory}" for memory in memories)

    @tool
    async def save_user_memory(memory: str) -> str:
        """Salva uma memoria curta e relevante para conversas futuras."""
        if not memory_is_allowed(memory):
            logger.warning("memory_tool_rejected tool=save_user_memory reason=invalid_memory")
            return "Memoria recusada: deve ser curta e nao pode conter credenciais."
        try:
            await memory_store.save(user_id, memory.strip())
        except Exception:
            logger.exception("memory_tool_failed tool=save_user_memory")
            return "Nao foi possivel salvar a memoria agora."
        logger.info("memory_tool_called tool=save_user_memory")
        return "Memoria salva para este usuario."

    return [recall_user_memories, save_user_memory]


TOOLS: list = []
