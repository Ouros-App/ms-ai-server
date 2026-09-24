import asyncio
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram

from app.core.config import settings

HTTP_REQUESTS = Counter(
    "ai_server_http_requests",
    "Total de requisicoes HTTP recebidas.",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "ai_server_http_request_duration_seconds",
    "Duracao das requisicoes HTTP em segundos.",
    ("method", "route"),
)
CHAT_REQUESTS = Counter(
    "ai_server_chat_requests",
    "Total de mensagens processadas pelo chat.",
    ("outcome",),
)
CHAT_DURATION = Histogram(
    "ai_server_chat_duration_seconds",
    "Tempo ponta a ponta para processar uma mensagem do Midas.",
    ("outcome",),
)
CHAT_IN_FLIGHT = Gauge(
    "ai_server_chat_in_flight",
    "Quantidade de mensagens do Midas em processamento neste processo.",
)
CHAT_AGENTS = Counter(
    "ai_server_chat_agent_usage",
    "Quantidade de agentes registrados nas respostas do chat.",
    ("agent",),
)
CHAT_TOOLS = Counter(
    "ai_server_chat_tool_usage",
    "Quantidade de tools usadas pelo chat.",
    ("tool",),
)
CHAT_ROUTES = Counter(
    "ai_server_chat_route_usage",
    "Quantidade de rotas selecionadas pelo chat por origem.",
    ("route", "source"),
)
CHAT_PENDING = Counter(
    "ai_server_chat_pending_tasks",
    "Quantidade de tarefas que terminaram o turno aguardando dado adicional.",
    ("route",),
)
LLM_REQUESTS = Counter(
    "ai_server_llm_requests",
    "Chamadas a modelos de linguagem observadas pelo Midas.",
    ("profile", "model", "outcome"),
)
LLM_DURATION = Histogram(
    "ai_server_llm_request_duration_seconds",
    "Duracao das chamadas a modelos de linguagem.",
    ("profile", "model"),
)
LLM_IN_FLIGHT = Gauge(
    "ai_server_llm_in_flight",
    "Chamadas LLM atualmente em processamento neste processo.",
)
LLM_INPUT_TOKENS = Counter(
    "ai_server_llm_input_tokens",
    "Tokens de entrada reportados pelos provedores de LLM, incluindo os servidos por cache.",
    ("profile", "model"),
)
LLM_CACHED_INPUT_TOKENS = Counter(
    "ai_server_llm_cached_input_tokens",
    "Subconjunto dos tokens de entrada servido por cache, quando reportado.",
    ("profile", "model"),
)
LLM_OUTPUT_TOKENS = Counter(
    "ai_server_llm_output_tokens",
    "Tokens de saida reportados pelos provedores de LLM.",
    ("profile", "model"),
)
MCP_REQUESTS = Counter(
    "ai_server_mcp_requests",
    "Chamadas do AI Server ao Knowledge MCP.",
    ("tool", "outcome"),
)
MCP_DURATION = Histogram(
    "ai_server_mcp_request_duration_seconds",
    "Tempo de resposta observado pelo AI Server ao chamar o Knowledge MCP.",
    ("tool",),
)
MCP_IN_FLIGHT = Gauge(
    "ai_server_mcp_in_flight",
    "Chamadas ao Knowledge MCP atualmente em processamento.",
)

_ALLOWED_ROUTE_SOURCES = {
    "cancelled",
    "context",
    "deterministic",
    "explicit",
    "greeting",
    "guardrail",
    "identity",
    "model",
    "no_model",
    "pending",
    "router_error",
}
_ALLOWED_PROFILES = {"fast", "powerful"}
_ALLOWED_MODELS = {
    settings.groq_fast_model,
    settings.groq_model,
    settings.nvidia_nim_fast_model,
    settings.nvidia_nim_model,
}
_ALLOWED_MCP_TOOLS = {
    "search_knowledge",
    "qdrant_status",
    "postgres_status",
    "get_user_context",
    "get_consumption_summary",
    "prepare_resource_import",
    "import_user_resource_records",
}


def _safe_route_source(source: object) -> str:
    return (
        source
        if isinstance(source, str) and source in _ALLOWED_ROUTE_SOURCES
        else "unknown"
    )


def _safe_profile(profile: object) -> str:
    return profile if isinstance(profile, str) and profile in _ALLOWED_PROFILES else "unknown"


def _safe_model(model: object) -> str:
    return model if isinstance(model, str) and model in _ALLOWED_MODELS else "unknown"


def _safe_mcp_tool(tool: object) -> str:
    return tool if isinstance(tool, str) and tool in _ALLOWED_MCP_TOOLS else "unknown"


def _response_model_name(response: object) -> str:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        for key in ("model_name", "model", "model_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return _safe_model(value)
    return "unknown"


def _usage_dict(response: object) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if isinstance(usage, dict):
        return usage

    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return {}
    token_usage = metadata.get("token_usage") or metadata.get("usage")
    return token_usage if isinstance(token_usage, dict) else {}


def _token_count(usage: dict, *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _cached_input_tokens(usage: dict) -> int:
    details = usage.get("input_token_details")
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict):
        return _token_count(
            details,
            "cache_read",
            "cached_tokens",
            "cache_read_tokens",
        )
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        return _token_count(prompt_details, "cached_tokens")
    return 0


def observe_http_request(
    method: str,
    route: str,
    status: int | str,
    duration_seconds: float,
) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_REQUEST_DURATION.labels(method, route).observe(duration_seconds)


def observe_chat_result(outcome: str, agents: list[str], tools: list[str]) -> None:
    CHAT_REQUESTS.labels(outcome).inc()
    for agent in agents:
        CHAT_AGENTS.labels(agent).inc()
    for tool in tools:
        CHAT_TOOLS.labels(tool).inc()


def observe_chat_duration(outcome: str, duration_seconds: float) -> None:
    CHAT_DURATION.labels(outcome).observe(max(duration_seconds, 0.0))


def chat_started() -> None:
    CHAT_IN_FLIGHT.inc()


def chat_finished() -> None:
    CHAT_IN_FLIGHT.dec()


def observe_chat_routing(
    routes: list[str],
    source: object,
    pending_routes: list[str],
) -> None:
    """Record only bounded routing labels to keep Prometheus cardinality stable."""
    route_source = _safe_route_source(source)
    for route in routes:
        if route:
            CHAT_ROUTES.labels(route, route_source).inc()
    for route in pending_routes:
        if route:
            CHAT_PENDING.labels(route).inc()


async def observed_llm_ainvoke(model, messages: list, profile: str):
    """Invoke an LLM and record provider-reported usage without logging prompts."""
    safe_profile = _safe_profile(profile)
    started_at = perf_counter()
    LLM_IN_FLIGHT.inc()
    try:
        response = await model.ainvoke(messages)
    except asyncio.CancelledError:
        duration = perf_counter() - started_at
        LLM_REQUESTS.labels(safe_profile, "unknown", "cancelled").inc()
        LLM_DURATION.labels(safe_profile, "unknown").observe(duration)
        raise
    except Exception:
        duration = perf_counter() - started_at
        LLM_REQUESTS.labels(safe_profile, "unknown", "error").inc()
        LLM_DURATION.labels(safe_profile, "unknown").observe(duration)
        raise
    finally:
        LLM_IN_FLIGHT.dec()

    duration = perf_counter() - started_at
    model_name = _response_model_name(response)
    usage = _usage_dict(response)
    input_tokens = _token_count(usage, "input_tokens", "prompt_tokens")
    output_tokens = _token_count(usage, "output_tokens", "completion_tokens")
    cached_tokens = min(_cached_input_tokens(usage), input_tokens)

    LLM_REQUESTS.labels(safe_profile, model_name, "success").inc()
    LLM_DURATION.labels(safe_profile, model_name).observe(duration)
    if input_tokens:
        LLM_INPUT_TOKENS.labels(safe_profile, model_name).inc(input_tokens)
    if cached_tokens:
        LLM_CACHED_INPUT_TOKENS.labels(safe_profile, model_name).inc(cached_tokens)
    if output_tokens:
        LLM_OUTPUT_TOKENS.labels(safe_profile, model_name).inc(output_tokens)
    return response


def mcp_call_started() -> None:
    MCP_IN_FLIGHT.inc()


def observe_mcp_call(tool: str, outcome: str, duration_seconds: float) -> None:
    safe_tool = _safe_mcp_tool(tool)
    MCP_IN_FLIGHT.dec()
    MCP_REQUESTS.labels(safe_tool, outcome).inc()
    MCP_DURATION.labels(safe_tool).observe(max(duration_seconds, 0.0))
