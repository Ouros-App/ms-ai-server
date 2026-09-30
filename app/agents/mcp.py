import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from time import monotonic
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.token_exchange import (
    MCPTokenExchangeError,
    exchange_mcp_access_token,
    exchange_telemetry_access_token,
)
from app.debug_ui.trace import trace_event

logger = logging.getLogger(__name__)

MCP_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "faq": frozenset(
        {"search_knowledge", "get_user_context", "create_custom_dashboard"}
    ),
    "sustainability": frozenset(
        {
            "search_knowledge",
            "get_user_context",
            "get_consumption_summary",
            "create_custom_dashboard",
        }
    ),
    "ranking": frozenset(
        {"search_knowledge", "get_user_context", "create_custom_dashboard"}
    ),
    "support": frozenset({"search_knowledge", "get_user_context"}),
}
MCP_USER_SCOPED_TOOLS = frozenset(
    {"get_user_context", "get_consumption_summary", "create_custom_dashboard"}
)
_CUSTOM_DASHBOARD_RENDER_TYPES = frozenset(
    {"indicator", "bar", "line", "pie", "donut", "histogram"}
)
MCP_TOOLS_CACHE_MAX_ENTRIES = 256
DEFAULT_CONSUMPTION_PERIOD_DAYS = 30
MAX_CONSUMPTION_PERIOD_DAYS = 366
_FORWARDED_ACCESS_TOKEN: ContextVar[str | None] = ContextVar(
    "mcp_forwarded_access_token",
    default=None,
)
_MCP_VISUALIZATIONS: ContextVar[list[dict] | None] = ContextVar(
    "mcp_chat_visualizations",
    default=None,
)


@contextmanager
def capture_mcp_visualizations():
    """Collect dashboard artifacts created during one chat request."""
    visualizations: list[dict] = []
    marker = _MCP_VISUALIZATIONS.set(visualizations)
    try:
        yield visualizations
    finally:
        _MCP_VISUALIZATIONS.reset(marker)


def _record_visualization(visualization: dict) -> None:
    current = _MCP_VISUALIZATIONS.get()
    if current is not None and not current:
        current.append(visualization)


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
    """Represent a tool contract that accepts no model-provided arguments."""


class _ConsumptionSummaryArguments(BaseModel):
    """Validate the bounded period accepted by consumption summaries."""

    period_days: int = Field(
        default=DEFAULT_CONSUMPTION_PERIOD_DAYS,
        ge=1,
        le=MAX_CONSUMPTION_PERIOD_DAYS,
        description=(
            "Janela de consulta em dias, limitada pelo contrato da tool."
        ),
    )


class _CustomDashboardChartArguments(BaseModel):
    chart_id: str = Field(
        min_length=1,
        max_length=64,
        description="ID de um gráfico permitido pelo catálogo.",
    )
    render_as: Literal[
        "auto",
        "indicator",
        "bar",
        "line",
        "pie",
        "donut",
        "histogram",
    ] = Field(
        default="auto",
        description=(
            "Tipo pedido pelo usuário quando compatível. Histogramas usam séries "
            "numéricas; pizza só está disponível em goal-status. Use auto quando "
            "nenhum tipo for pedido."
        ),
    )


class _CustomDashboardArguments(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    charts: list[_CustomDashboardChartArguments] = Field(min_length=1, max_length=4)


class MCPToolProvider:
    """Load MCP tools per specialist without exposing credentials to the model."""

    server_name = "midas"

    def __init__(
        self,
        url: str | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        """Initialize an MCP provider with a bounded per-token tool cache."""
        self.url = url
        self.cache_ttl_seconds = cache_ttl_seconds
        self._tools_cache: dict[str, tuple[list, float]] = {}
        self._tools_cache_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> "MCPToolProvider":
        """Build the provider from the process MCP configuration."""
        return cls(
            url=settings.mcp_url,
            cache_ttl_seconds=settings.mcp_tools_cache_ttl_seconds,
        )

    def _token_for(self) -> str | None:
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

    async def _load_tools(self, token: str, *, include_dashboard: bool = False) -> list:
        """Exchange a user token, discover MCP tools, and cache the result."""
        now = monotonic()
        cache_key = f"{self._token_cache_key(token)}:{int(include_dashboard)}"
        cached = self._tools_cache.get(cache_key)
        if cached and now - cached[1] < self.cache_ttl_seconds:
            return cached[0]

        async with self._tools_cache_lock:
            now = monotonic()
            cached = self._tools_cache.get(cache_key)
            if cached and now - cached[1] < self.cache_ttl_seconds:
                return cached[0]
            self._prune_tools_cache(now)

            logger.info("mcp_tools_load_started server=%s", self.server_name)
            if include_dashboard:
                delegated_token, telemetry_token = await asyncio.gather(
                    exchange_mcp_access_token(token),
                    exchange_telemetry_access_token(token),
                )
                headers = {
                    "Authorization": f"Bearer {delegated_token}",
                    "X-Ouros-Telemetry-Token": f"Bearer {telemetry_token}",
                }
            else:
                delegated_token = await exchange_mcp_access_token(token)
                headers = {"Authorization": f"Bearer {delegated_token}"}

            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    self.server_name: {
                        "transport": "http",
                        "url": self.url,
                        "headers": headers,
                    },
                },
                handle_tool_errors=True,
            )
            tools = await client.get_tools(server_name=self.server_name)
            self._tools_cache[cache_key] = (tools, now)
            logger.info(
                "mcp_tools_load_succeeded server=%s discovered=%d",
                self.server_name,
                len(tools),
            )
            return tools

    async def tools_for(
        self,
        agent_name: str,
        *,
        dashboard_requested: bool = False,
        dashboard_period_days: int = DEFAULT_CONSUMPTION_PERIOD_DAYS,
    ) -> list:
        """Return only MCP tools authorized for one specialist."""

        allowed = MCP_TOOL_ALLOWLIST.get(agent_name, frozenset())
        token = self._token_for()
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
            tools = await self._load_tools(
                token,
                include_dashboard=dashboard_requested,
            )
        except MCPTokenExchangeError as error:
            logger.warning(
                "mcp_token_exchange_unavailable agent=%s reason=%s status=%s",
                agent_name,
                error.reason,
                error.status_code if error.status_code is not None else "none",
            )
            trace_event(
                "mcp.token_exchange_failed",
                agent=agent_name,
                reason=error.reason,
                status=error.status_code,
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
            if tool.name == "create_custom_dashboard" and not dashboard_requested:
                continue
            if tool.name not in MCP_USER_SCOPED_TOOLS:
                selected.append(tool)
                continue
            selected.append(
                self._bind_user_tool(
                    tool,
                    dashboard_period_days=dashboard_period_days,
                )
            )
        logger.info("mcp_tools_loaded agent=%s count=%d", agent_name, len(selected))
        return selected

    def _bind_user_tool(
        self,
        tool,
        *,
        dashboard_period_days: int = DEFAULT_CONSUMPTION_PERIOD_DAYS,
    ) -> StructuredTool:
        """Bind authenticated identity without exposing identifiers to the model."""

        description = getattr(tool, "description", None) or tool.name
        if tool.name == "get_user_context":

            async def invoke() -> object:
                result = await self._invoke_remote_tool(tool, {})
                return self._filter_user_context(result)

            args_schema = _NoArguments
        elif tool.name == "get_consumption_summary":

            async def invoke(
                period_days: int = DEFAULT_CONSUMPTION_PERIOD_DAYS,
            ) -> object:
                result = await self._invoke_remote_tool(
                    tool,
                    {"period_days": period_days},
                )
                return self._filter_consumption_summary(
                    result,
                    expected_period_days=period_days,
                )

            args_schema = _ConsumptionSummaryArguments
        elif tool.name == "create_custom_dashboard":

            async def invoke(
                title: str,
                charts: list[dict[str, str]],
                period_days: int = dashboard_period_days,
            ) -> object:
                chart_selections = [
                    chart.model_dump() if isinstance(chart, BaseModel) else chart
                    for chart in charts
                ]
                chart_ids = [chart["chart_id"] for chart in chart_selections]
                result = await self._invoke_remote_tool(
                    tool,
                    {
                        "title": title,
                        "charts": chart_selections,
                        "period_days": period_days,
                    },
                )
                dashboard = self._filter_custom_dashboard(
                    result,
                    expected_chart_ids=chart_ids,
                )
                _record_visualization(dashboard)
                return {
                    "title": dashboard["title"],
                    "charts": [
                        {"id": item["id"], "title": item["title"]}
                        for item in dashboard["charts"]
                    ],
                }

            args_schema = _CustomDashboardArguments
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

    @classmethod
    def _filter_custom_dashboard(
        cls,
        result: object,
        *,
        expected_chart_ids: list[str],
    ) -> dict:
        """Keep only the bounded, expected visual payload for the chat client."""
        dashboard = cls._require_decoded_result(
            result,
            tool_name="create_custom_dashboard",
        )
        title = dashboard.get("title")
        charts = dashboard.get("charts")
        if (
            not isinstance(title, str)
            or not title.strip()
            or len(title) > 120
            or not isinstance(charts, list)
            or len(charts) != len(expected_chart_ids)
        ):
            raise MCPToolResultError("invalid custom dashboard result")

        chart_by_id = {
            item.get("id"): item
            for item in charts
            if isinstance(item, dict)
        }
        if set(chart_by_id) != set(expected_chart_ids):
            raise MCPToolResultError("custom dashboard chart ids do not match")

        filtered_charts = []
        html_chars = 0
        for chart_id in expected_chart_ids:
            item = chart_by_id[chart_id]
            chart_title = item.get("title")
            render_as = item.get("render_as")
            html = item.get("html")
            if (
                not isinstance(chart_title, str)
                or not chart_title.strip()
                or not isinstance(render_as, str)
                or render_as not in _CUSTOM_DASHBOARD_RENDER_TYPES
                or not isinstance(html, str)
                or not html
                or len(html) > 1_500_000
            ):
                raise MCPToolResultError("invalid custom dashboard chart")
            html_chars += len(html)
            filtered_charts.append(
                {
                    "id": chart_id,
                    "title": chart_title[:200],
                    "render_as": render_as,
                    "html": html,
                }
            )
        if html_chars > 2_000_000:
            raise MCPToolResultError("custom dashboard result is too large")
        return {
            "type": "ouros_dashboard",
            "title": title.strip(),
            "charts": filtered_charts,
        }

    @staticmethod
    async def _invoke_remote_tool(tool, arguments: dict[str, object]) -> object:
        """Invoke an MCP LangChain tool while preserving its structured artifact."""
        tool_call = {
            "type": "tool_call",
            "id": f"mcp-bound-{tool.name}",
            "name": tool.name,
            "args": arguments,
        }
        return await tool.ainvoke(tool_call)

    @staticmethod
    def _require_decoded_result(
        result: object,
        *,
        tool_name: str,
        expected_period_days: int | None = None,
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
            expected_period_days=expected_period_days,
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
        expected_period_days: int | None = None,
    ) -> bool:
        """Validate the minimum trusted shape returned by user-scoped MCP tools."""
        if tool_name == "get_user_context":
            return (
                isinstance(result.get("user_type"), str)
                and isinstance(result.get("profile"), dict)
                and isinstance(result.get("farms"), list)
                and isinstance(result.get("enterprises"), list)
            )

        if tool_name == "get_consumption_summary":
            farm_ids = result.get("farm_ids")
            period_days = result.get("period_days")
            return (
                isinstance(result.get("user_type"), str)
                and isinstance(result.get("user_id"), int)
                and not isinstance(result.get("user_id"), bool)
                and isinstance(period_days, int)
                and not isinstance(period_days, bool)
                and (
                    expected_period_days is None
                    or period_days == expected_period_days
                )
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
    _PUBLIC_FARM_CONTEXT_FIELDS = (
        "name",
        "area_property",
        "region",
        "poultry_capacity",
        "place",
        "state",
        "city",
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
        profile = result.get("profile")
        public_profile = {}
        if isinstance(profile, dict) and isinstance(profile.get("name"), str):
            public_profile["name"] = profile["name"]

        enterprises = result.get("enterprises")
        public_enterprises = []
        if isinstance(enterprises, list):
            public_enterprises = [
                {"name": item["name"]}
                for item in enterprises
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]

        farms = result.get("farms")
        public_farms = []
        if isinstance(farms, list):
            public_farms = [
                {
                    key: item[key]
                    for key in MCPToolProvider._PUBLIC_FARM_CONTEXT_FIELDS
                    if key in item
                }
                for item in farms
                if isinstance(item, dict)
            ]

        return MCPToolProvider._without_internal_ids(
            {
                "profile": public_profile,
                "enterprises": public_enterprises,
                "farms": public_farms,
            }
        )

    @staticmethod
    def _filter_consumption_summary(
        result: object,
        *,
        expected_period_days: int,
    ) -> dict:
        """Defense-in-depth filter for scoped aggregate consumption results."""

        result = MCPToolProvider._require_decoded_result(
            result,
            tool_name="get_consumption_summary",
            expected_period_days=expected_period_days,
        )
        authorized_ids = [
            item
            for item in result.get("farm_ids", [])
            if isinstance(item, int) and not isinstance(item, bool)
        ]
        if not authorized_ids:
            logger.warning("mcp_consumption_scope_denied")
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
