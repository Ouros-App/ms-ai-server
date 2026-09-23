import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from time import monotonic, perf_counter

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.metrics import mcp_call_started, observe_mcp_call
from app.core.token_exchange import MCPTokenExchangeError, exchange_mcp_access_token
from app.debug_ui.trace import trace_event

logger = logging.getLogger(__name__)

MCP_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "faq": frozenset({"search_knowledge", "get_user_context"}),
    "sustainability": frozenset(
        {"search_knowledge", "get_user_context", "get_consumption_summary"}
    ),
    "ranking": frozenset({"search_knowledge", "get_user_context"}),
    "support": frozenset({"search_knowledge", "get_user_context"}),
    "fallback": frozenset({"search_knowledge"}),
}
MCP_USER_SCOPED_TOOLS = frozenset(
    {"get_user_context", "get_user_farm_data", "get_consumption_summary"}
)
MCP_TOOLS_CACHE_MAX_ENTRIES = 256
_FORWARDED_ACCESS_TOKEN: ContextVar[str | None] = ContextVar(
    "mcp_forwarded_access_token",
    default=None,
)


@contextmanager
def forward_mcp_access_token(token: str | None):
    """Expose one validated Keycloak token only for the current request."""

    token_marker = _FORWARDED_ACCESS_TOKEN.set(token)
    try:
        yield
    finally:
        _FORWARDED_ACCESS_TOKEN.reset(token_marker)


class MCPToolResultError(RuntimeError):
    """Raised when an MCP tool result cannot be decoded safely."""


class _NoArguments(BaseModel):
    pass


class _FarmDataArguments(BaseModel):
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Quantidade maxima de registros por conjunto de dados.",
    )


class _ConsumptionSummaryArguments(BaseModel):
    period_days: int = Field(
        default=30,
        ge=1,
        le=366,
        description="Janela de consulta em dias, entre 1 e 366.",
    )


class MCPToolProvider:
    """Load MCP tools per specialist without exposing credentials to the model."""

    server_name = "midas"

    def __init__(
        self,
        url: str | None = None,
        resource_url: str | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.url = url
        self.resource_url = resource_url or url
        self.cache_ttl_seconds = cache_ttl_seconds
        self._tools_cache: dict[str, tuple[list, float]] = {}
        self._tools_cache_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> "MCPToolProvider":
        return cls(
            url=settings.mcp_url,
            resource_url=settings.mcp_resource_url,
            cache_ttl_seconds=settings.mcp_tools_cache_ttl_seconds,
        )

    def _token_for(self, _user_id: str) -> str | None:
        """Return only the validated Keycloak token forwarded by the API."""

        return _FORWARDED_ACCESS_TOKEN.get()

    @staticmethod
    def _token_cache_key(token: str) -> str:
        """Avoid retaining raw Bearer tokens as dictionary keys."""

        return sha256(token.encode()).hexdigest()

    def _prune_tools_cache(self, now: float) -> None:
        """Remove expired entries and keep the per-token tools cache bounded."""

        expired_keys = [
            key
            for key, (_, created_at) in self._tools_cache.items()
            if now - created_at >= self.cache_ttl_seconds
        ]
        for key in expired_keys:
            self._tools_cache.pop(key, None)

        while len(self._tools_cache) >= MCP_TOOLS_CACHE_MAX_ENTRIES:
            oldest_key = next(iter(self._tools_cache))
            self._tools_cache.pop(oldest_key, None)

    async def _load_tools(self, token: str) -> list:
        now = monotonic()
        cache_key = self._token_cache_key(token)
        cached = self._tools_cache.get(cache_key)
        if cached and now - cached[1] < self.cache_ttl_seconds:
            return cached[0]

        async with self._tools_cache_lock:
            now = monotonic()
            cached = self._tools_cache.get(cache_key)
            if cached and now - cached[1] < self.cache_ttl_seconds:
                return cached[0]
            self._prune_tools_cache(now)

            delegated_token = await exchange_mcp_access_token(token)

            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    self.server_name: {
                        "transport": "http",
                        "url": self.url,
                        "headers": {"Authorization": f"Bearer {delegated_token}"},
                    },
                },
                handle_tool_errors=True,
            )
            tools = await client.get_tools(server_name=self.server_name)
            self._tools_cache[cache_key] = (tools, now)
            return tools

    async def tools_for(
        self,
        agent_name: str,
        user_id: str,
        request_text: str | None = None,
    ) -> list:
        """Return only MCP tools authorized for one specialist."""

        allowed = MCP_TOOL_ALLOWLIST.get(agent_name, frozenset())
        token = self._token_for(user_id)
        if not self.url:
            trace_event(
                "mcp.tools_unavailable",
                agent=agent_name,
                reason="missing_url",
            )
            return []
        if not allowed:
            return []
        if not token:
            trace_event(
                "mcp.tools_unavailable",
                agent=agent_name,
                reason="missing_forwarded_token",
            )
            return []

        try:
            tools = await self._load_tools(token)
        except MCPTokenExchangeError as error:
            logger.warning(
                "mcp_token_exchange_unavailable agent=%s error=%s",
                agent_name,
                type(error).__name__,
            )
            trace_event(
                "mcp.token_exchange_failed",
                agent=agent_name,
                error=type(error).__name__,
            )
            return []
        except Exception as error:
            logger.exception("mcp_tools_load_failed agent=%s", agent_name)
            trace_event(
                "mcp.tools_load_failed",
                agent=agent_name,
                error=type(error).__name__,
            )
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
            selected.append(
                self._bind_user_tool(
                    tool,
                    numeric_user_id,
                    request_text=request_text,
                )
            )
        logger.info("mcp_tools_loaded agent=%s count=%d", agent_name, len(selected))
        return selected

    def _bind_user_tool(
        self,
        tool,
        user_id: int,
        request_text: str | None = None,
    ) -> StructuredTool:
        """Bind authenticated identity without exposing identifiers to the model."""

        description = getattr(tool, "description", None) or tool.name
        if tool.name == "get_user_context":

            async def invoke() -> object:
                result = await self._invoke_remote_tool(tool, {})
                return self._filter_user_context(result)

            args_schema = _NoArguments
        elif tool.name == "get_user_farm_data":

            async def invoke(limit: int = 20) -> object:
                result = await self._invoke_remote_tool(
                    tool,
                    {"limit": limit},
                )
                return self._filter_farm_data(result, user_id)

            args_schema = _FarmDataArguments
        elif tool.name == "get_consumption_summary":

            async def invoke(period_days: int = 30) -> object:
                result = await self._invoke_remote_tool(
                    tool,
                    {"period_days": period_days},
                )
                return self._filter_consumption_summary(result, user_id)

            args_schema = _ConsumptionSummaryArguments
        else:
            raise ValueError(f"unsupported user-scoped MCP tool: {tool.name}")

        if tool.name in MCP_USER_SCOPED_TOOLS:
            description = (
                f"{description} A identidade e as fazendas autorizadas sao resolvidas "
                "pelo backend a partir do JWT. Nunca solicite farm_id, user_id ou "
                "user_type ao usuario."
            )

        return StructuredTool.from_function(
            coroutine=invoke,
            name=tool.name,
            description=description,
            args_schema=args_schema,
        )

    @staticmethod
    async def _invoke_remote_tool(tool, arguments: dict[str, object]) -> object:
        """Invoke an MCP LangChain tool while preserving its structured artifact."""
        tool_call = {
            "type": "tool_call",
            "id": f"mcp-bound-{tool.name}",
            "name": tool.name,
            "args": arguments,
        }
        started_at = perf_counter()
        mcp_call_started()
        try:
            result = await tool.ainvoke(tool_call)
        except Exception:
            observe_mcp_call(
                tool.name,
                "error",
                perf_counter() - started_at,
            )
            raise
        observe_mcp_call(
            tool.name,
            "success",
            perf_counter() - started_at,
        )
        return result

    @staticmethod
    def _require_decoded_result(
        result: object,
        *,
        tool_name: str,
    ) -> dict:
        """Decode and validate the contract for a structured MCP tool result."""
        if isinstance(result, ToolMessage) and result.status == "error":
            message = MCPToolProvider._tool_message_text(result)
            trace_event(
                "mcp.tool_error",
                tool=tool_name,
                error_chars=len(message),
            )
            raise MCPToolResultError(
                f"remote MCP tool {tool_name} returned an error"
            )

        decoded = MCPToolProvider._decode_tool_result(result)
        if decoded is not None and MCPToolProvider._result_contract_is_valid(
            decoded,
            tool_name=tool_name,
        ):
            return decoded

        trace_event(
            "mcp.tool_result_invalid",
            tool=tool_name,
            result_type=type(result).__name__,
        )
        raise MCPToolResultError(
            f"invalid structured result returned by MCP tool {tool_name}"
        )

    @staticmethod
    def _tool_message_text(result: ToolMessage) -> str:
        """Return a bounded human-readable error summary from a ToolMessage."""
        content = result.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ]
            text = " ".join(part.strip() for part in parts if part.strip())
        else:
            text = ""

        compact = " ".join(text.split())
        return compact[:500] if compact else "MCP tool returned an unspecified error."

    @staticmethod
    def _result_contract_is_valid(
        result: dict,
        *,
        tool_name: str,
    ) -> bool:
        """Validate the minimum trusted shape returned by user-scoped MCP tools."""
        if tool_name == "get_user_farm_data":
            farm_ids = result.get("farm_ids")
            data = result.get("data")
            return (
                isinstance(farm_ids, list)
                and all(
                    isinstance(farm_id, int) and not isinstance(farm_id, bool)
                    for farm_id in farm_ids
                )
                and isinstance(data, dict)
            )

        if tool_name == "get_user_context":
            return (
                isinstance(result.get("user_type"), str)
                and isinstance(result.get("user_id"), int)
                and not isinstance(result.get("user_id"), bool)
                and isinstance(result.get("profile"), dict)
                and isinstance(result.get("farms"), list)
                and isinstance(result.get("enterprises"), list)
            )

        if tool_name == "get_consumption_summary":
            farm_ids = result.get("farm_ids")
            return (
                isinstance(result.get("user_type"), str)
                and isinstance(result.get("user_id"), int)
                and not isinstance(result.get("user_id"), bool)
                and isinstance(result.get("period_days"), int)
                and not isinstance(result.get("period_days"), bool)
                and isinstance(farm_ids, list)
                and all(
                    isinstance(farm_id, int) and not isinstance(farm_id, bool)
                    for farm_id in farm_ids
                )
                and isinstance(result.get("summaries"), list)
            )

        return isinstance(result, dict)

    _INTERNAL_ID_KEYS = frozenset(
        {
            "id",
            "id_farm",
            "farm_id",
            "user_id",
            "id_user",
            "id_enterprise",
            "enterprise_id",
        }
    )

    @staticmethod
    def _without_internal_ids(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: MCPToolProvider._without_internal_ids(item)
                for key, item in value.items()
                if key not in MCPToolProvider._INTERNAL_ID_KEYS
            }
        if isinstance(value, list):
            return [
                MCPToolProvider._without_internal_ids(item)
                for item in value
            ]
        return value

    @staticmethod
    def _filter_user_context(result: object) -> dict:
        """Hide backend identity and object IDs before context reaches the model."""
        result = MCPToolProvider._require_decoded_result(
            result,
            tool_name="get_user_context",
        )
        return MCPToolProvider._without_internal_ids(
            {
                "profile": result.get("profile", {}),
                "enterprises": result.get("enterprises", []),
                "farms": result.get("farms", []),
            }
        )

    @staticmethod
    def _filter_farm_data(
        result: object,
        user_id: int,
    ) -> dict:
        """Return only records for farms authorized by the MCP response."""

        result = MCPToolProvider._require_decoded_result(
            result,
            tool_name="get_user_farm_data",
        )

        authorized_ids = [
            item
            for item in result.get("farm_ids", [])
            if isinstance(item, int)
        ]
        if not authorized_ids:
            logger.warning("mcp_farm_scope_denied user_id=%s", user_id)
            return {
                "user_type": result.get("user_type"),
                "user_id": user_id,
                "authorized": False,
                "reason": "no_farm_scope",
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
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get(key) in authorized_ids
                ]
        return {
            "user_type": result.get("user_type"),
            "user_id": user_id,
            "authorized": True,
            "data": data,
        }

    @staticmethod
    def _filter_consumption_summary(
        result: object,
        user_id: int,
    ) -> dict:
        """Defense-in-depth filter for scoped aggregate consumption results."""

        result = MCPToolProvider._require_decoded_result(
            result,
            tool_name="get_consumption_summary",
        )
        authorized_ids = [
            item
            for item in result.get("farm_ids", [])
            if isinstance(item, int) and not isinstance(item, bool)
        ]
        if not authorized_ids:
            logger.warning("mcp_consumption_scope_denied user_id=%s", user_id)
            return {
                "authorized": False,
                "reason": "no_farm_scope",
                "period_days": result.get("period_days"),
                "water_unit": result.get("water_unit"),
                "energy_unit": result.get("energy_unit"),
                "summaries": [],
            }

        summaries = []
        for row in result.get("summaries", []):
            if not isinstance(row, dict) or row.get("id_farm") not in authorized_ids:
                continue
            summaries.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "id_farm"
                }
            )
        return {
            "authorized": True,
            "period_days": result.get("period_days"),
            "water_unit": result.get("water_unit"),
            "energy_unit": result.get("energy_unit"),
            "summaries": summaries,
        }

    @staticmethod
    def _decode_tool_result(result: object) -> dict | None:
        """Decode MCP structured content across LangChain adapter output shapes."""
        if isinstance(result, ToolMessage):
            artifact_result = MCPToolProvider._decode_tool_result(result.artifact)
            if artifact_result is not None:
                return artifact_result
            return MCPToolProvider._decode_tool_result(result.content)

        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            artifact_result = MCPToolProvider._decode_tool_result(artifact)
            if artifact_result is not None:
                return artifact_result
            return MCPToolProvider._decode_tool_result(content)

        if isinstance(result, dict):
            for key in ("structured_content", "structuredContent"):
                structured = result.get(key)
                if isinstance(structured, dict):
                    return structured

            artifact = result.get("artifact")
            if artifact is not None:
                artifact_result = MCPToolProvider._decode_tool_result(artifact)
                if artifact_result is not None:
                    return artifact_result

            if (
                result.get("type") == "text"
                and isinstance(result.get("text"), str)
            ):
                return MCPToolProvider._decode_tool_result(result["text"])

            return result

        if isinstance(result, str):
            try:
                decoded = json.loads(result)
            except json.JSONDecodeError:
                return None
            return MCPToolProvider._decode_tool_result(decoded)

        if isinstance(result, list):
            for item in result:
                if (
                    isinstance(item, dict)
                    and "type" in item
                    and item.get("type") != "text"
                ):
                    continue
                decoded = MCPToolProvider._decode_tool_result(item)
                if decoded is not None:
                    return decoded
            text = "".join(
                item.get("text", "")
                for item in result
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
            return MCPToolProvider._decode_tool_result(text) if text else None

        artifact = getattr(result, "artifact", None)
        if artifact is not None:
            decoded = MCPToolProvider._decode_tool_result(artifact)
            if decoded is not None:
                return decoded

        content = getattr(result, "content", None)
        if content is not None and content is not result:
            decoded = MCPToolProvider._decode_tool_result(content)
            if decoded is not None:
                return decoded

        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except (TypeError, ValueError):
                return None
            if dumped is not result:
                return MCPToolProvider._decode_tool_result(dumped)

        return None

