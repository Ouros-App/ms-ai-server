import asyncio
import logging
from datetime import datetime, timedelta, timezone
from time import monotonic

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

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
            async def invoke(limit: int = 20) -> object:
                return await tool.ainvoke(
                    {
                        "user_type": self.user_type,
                        "user_id": user_id,
                        "limit": limit,
                    }
                )

            args_schema = _FarmDataArguments

        return StructuredTool.from_function(
            coroutine=invoke,
            name=tool.name,
            description=description,
            args_schema=args_schema,
        )
