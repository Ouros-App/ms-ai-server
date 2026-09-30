from prometheus_client import Counter, Histogram

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
JEV_REQUESTS = Counter(
    "midas_jev_requests",
    "Requests sent to Jev by bounded status and decision type.",
    ("status", "decision_type"),
)
JEV_REQUEST_DURATION = Histogram(
    "midas_jev_request_duration_seconds",
    "Time spent waiting for Jev decision responses.",
    ("decision_type",),
)
JEV_INPUT_TOKENS = Counter(
    "midas_jev_input_tokens",
    "Input tokens reported by Jev.",
)
JEV_OUTPUT_TOKENS = Counter(
    "midas_jev_output_tokens",
    "Output tokens reported by Jev.",
)
JEV_COST_USD = Counter(
    "midas_jev_cost_usd",
    "Estimated Jev request cost in USD from configured token rates.",
)
JEV_FALLBACKS = Counter(
    "midas_jev_fallback",
    "Jev decisions that fell back to the existing Midas router.",
    ("reason",),
)
JEV_ROUTER_AGREEMENT = Counter(
    "midas_jev_router_agreement",
    "Shadow-mode Jev route agreement with the active Midas route.",
    ("agreement",),
)

_ALLOWED_ROUTE_SOURCES = {
    "cancelled",
    "context",
    "deterministic",
    "explicit",
    "greeting",
    "guardrail",
    "identity",
    "jev",
    "model",
    "no_model",
    "pending",
    "router_error",
}


def _safe_route_source(source: object) -> str:
    return source if isinstance(source, str) and source in _ALLOWED_ROUTE_SOURCES else "unknown"


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


def observe_jev_request(
    status: str,
    duration_seconds: float,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    fallback_reason: str | None = None,
) -> None:
    """Record one bounded Jev request without exposing state or credentials."""
    JEV_REQUESTS.labels(status, "combined").inc()
    JEV_REQUEST_DURATION.labels("combined").observe(duration_seconds)
    if input_tokens:
        JEV_INPUT_TOKENS.inc(input_tokens)
    if output_tokens:
        JEV_OUTPUT_TOKENS.inc(output_tokens)
    if cost_usd > 0:
        JEV_COST_USD.inc(cost_usd)
    if fallback_reason:
        JEV_FALLBACKS.labels(fallback_reason).inc()


def observe_jev_router_agreement(agreement: bool) -> None:
    """Record whether Jev and the active router selected the same agent."""
    JEV_ROUTER_AGREEMENT.labels(str(agreement).lower()).inc()
