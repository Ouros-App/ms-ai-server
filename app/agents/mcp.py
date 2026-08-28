import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from time import monotonic

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

MCP_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "faq": frozenset({"search_knowledge", "get_user_context"}),
    "sustainability": frozenset({"search_knowledge", "get_user_context", "get_user_farm_data"}),
    "ranking": frozenset({"get_user_context", "get_user_farm_data", "postgres_status"}),
    "support": frozenset({"search_knowledge", "get_user_context"}),
    "fallback": frozenset({"search_knowledge"}),
}
MCP_USER_SCOPED_TOOLS = frozenset({"get_user_context", "get_user_farm_data"})


class _NoArguments(BaseModel):
    pass


class _FarmDataArguments(BaseModel):
    farm_id: int | None = Field(
        default=None,
        gt=0,
        description="Identificador interno retornado pelo contexto autorizado",
    )
    limit: int = 20


class MCPToolProvider:
    """Carrega tools MCP por especialista sem expor o cliente ao sintetizador."""

    server_name = "midas"

    def __init__(
        self,
        url: str | None = None,
        access_token: str | None = None,
        jwt_secret: str | None = None,
        issuer_url: str | None = None,
        resource_url: str | None = None,
        user_type: str = "farm_owner",
        jwt_ttl_seconds: int = 300,
    ) -> None:
        self.url = url
        self.access_token = access_token
        self.jwt_secret = jwt_secret
        self.issuer_url = issuer_url
        self.resource_url = resource_url or url
        self.user_type = user_type
        self.jwt_ttl_seconds = jwt_ttl_seconds
        self._token_cache: dict[str, tuple[str, float]] = {}
        self._tools_cache: dict[str, tuple[list, float]] = {}
        self._tools_cache_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> "MCPToolProvider":
        return cls(
            url=settings.mcp_url,
            access_token=(
                settings.mcp_access_token.get_secret_value()
                if settings.mcp_access_token
                else None
            ),
            jwt_secret=(
                settings.mcp_jwt_secret.get_secret_value()
                if settings.mcp_jwt_secret
                else None
            ),
            issuer_url=settings.mcp_jwt_issuer_url,
            resource_url=settings.mcp_resource_url,
            user_type=settings.mcp_user_type,
            jwt_ttl_seconds=settings.mcp_jwt_ttl_seconds,
        )

    def _token_for(self, user_id: str) -> str | None:
        if self.access_token:
            return self.access_token
        if not self.jwt_secret:
            return None
        now_monotonic = monotonic()
        cached_token = self._token_cache.get(user_id)
        if cached_token and now_monotonic - cached_token[1] < self.jwt_ttl_seconds:
            return cached_token[0]
        try:
            numeric_user_id = int(user_id)
        except (TypeError, ValueError):
            logger.warning("mcp_tools_skipped reason=non_numeric_user_id")
            return None
        if numeric_user_id <= 0:
            return None

        import jwt

        now = datetime.now(timezone.utc)
        claims = {
            "sub": str(numeric_user_id),
            "user_type": self.user_type,
            "iss": self.issuer_url,
            "aud": self.resource_url,
            "iat": now,
            "exp": now + timedelta(seconds=self.jwt_ttl_seconds),
        }
        token = jwt.encode(claims, self.jwt_secret, algorithm="HS256")
        self._token_cache[user_id] = (token, now_monotonic)
        return token

    async def _load_tools(self, token: str) -> list:
        now = monotonic()
        cached = self._tools_cache.get(token)
        if cached and now - cached[1] < self.jwt_ttl_seconds:
            return cached[0]

        async with self._tools_cache_lock:
            now = monotonic()
            cached = self._tools_cache.get(token)
            if cached and now - cached[1] < self.jwt_ttl_seconds:
                return cached[0]

            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    self.server_name: {
                        "transport": "http",
                        "url": self.url,
                        "headers": {"Authorization": f"Bearer {token}"},
                    },
                },
                handle_tool_errors=True,
            )
            tools = await client.get_tools(server_name=self.server_name)
            self._tools_cache[token] = (tools, now)
            return tools

    async def tools_for(self, agent_name: str, user_id: str) -> list:
        """Retorna apenas as tools permitidas para o agente solicitado."""
        allowed = MCP_TOOL_ALLOWLIST.get(agent_name, frozenset())
        token = self._token_for(user_id)
        if not self.url or not allowed or not token:
            return []

        try:
            tools = await self._load_tools(token)
        except Exception:
            logger.exception("mcp_tools_load_failed agent=%s", agent_name)
            return []

        selected = []
        for tool in tools:
            if tool.name not in allowed:
                continue
            if tool.name not in MCP_USER_SCOPED_TOOLS:
                selected.append(tool)
                continue
            try:
                numeric_user_id = int(user_id)
            except (TypeError, ValueError):
                logger.warning("mcp_tools_skipped reason=non_numeric_user_id")
                continue
            selected.append(self._bind_user_tool(tool, numeric_user_id))
        logger.info("mcp_tools_loaded agent=%s count=%d", agent_name, len(selected))
        return selected

    def _bind_user_tool(self, tool, user_id: int) -> StructuredTool:
        """Vincula a identidade autenticada sem expor IDs ao modelo."""
        description = getattr(tool, "description", None) or tool.name
        if tool.name == "get_user_context":
            async def invoke() -> object:
                return await tool.ainvoke(
                    {"user_type": self.user_type, "user_id": user_id}
                )

            args_schema = _NoArguments
        else:
            async def invoke(farm_id: int | None = None, limit: int = 20) -> object:
                if farm_id is None:
                    return self._scope_denied(user_id)
                result = await tool.ainvoke(
                    {
                        "user_type": self.user_type,
                        "user_id": user_id,
                        "limit": limit,
                    }
                )
                return self._filter_farm_data(result, user_id, farm_id)

            args_schema = _FarmDataArguments

            description = (
                f"{description} Antes de usar, consulte get_user_context e use somente "
                "um farm_id retornado para este usuario. Nao trate nome ou ID citado "
                "na mensagem como prova de acesso."
            )

        return StructuredTool.from_function(
            coroutine=invoke,
            name=tool.name,
            description=description,
            args_schema=args_schema,
        )

    @staticmethod
    def _filter_farm_data(result: object, user_id: int, farm_id: int) -> dict:
        """Retorna somente registros da fazenda autorizada solicitada."""
        result = MCPToolProvider._decode_tool_result(result)
        if result is None:
            return MCPToolProvider._scope_denied(user_id)

        authorized_ids = result.get("farm_ids", [])
        if farm_id not in authorized_ids:
            logger.warning("mcp_farm_scope_denied user_id=%s farm_id=%s", user_id, farm_id)
            return {
                "user_type": result.get("user_type"),
                "user_id": user_id,
                "authorized": False,
                "data": {},
            }

        raw_data = result.get("data", {})
        data = {}
        if isinstance(raw_data, dict):
            for name, rows in raw_data.items():
                if not isinstance(rows, list):
                    continue
                key = "id" if name == "farms" else "id_farm"
                data[name] = [
                    row for row in rows
                    if isinstance(row, dict) and row.get(key) == farm_id
                ]
        return {
            "user_type": result.get("user_type"),
            "user_id": user_id,
            "authorized": True,
            "data": data,
        }

    @staticmethod
    def _decode_tool_result(result: object) -> dict | None:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                decoded = json.loads(result)
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, dict) else None
        if isinstance(result, list):
            text = "".join(
                item.get("text", "")
                for item in result
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
            return MCPToolProvider._decode_tool_result(text)
        return None

    @staticmethod
    def _scope_denied(user_id: int) -> dict:
        return {"user_id": user_id, "authorized": False, "data": {}}
