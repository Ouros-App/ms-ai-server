from langchain_core.callbacks import BaseCallbackHandler
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

AGENT_EXECUTION_DURATION = Histogram(
    "ai_server_agent_execution_duration_seconds",
    "Tempo de execucao de cada agente do chat em segundos.",
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
LLM_REQUESTS = Counter(
    "ai_server_llm_requests",
    "Total LangChain model calls by provider, profile, and outcome.",
    ("provider", "profile", "outcome"),
)
LLM_TOKENS = Counter(
    "ai_server_llm_tokens",
    "Input and output tokens reported by LangChain model responses.",
    ("provider", "profile", "direction"),
)
LLM_COST_USD = Counter(
    "ai_server_llm_cost_usd",
    "Estimated OpenAI model cost from reported token usage and published rates.",
    ("provider", "profile"),
)
LLM_FALLBACKS = Counter(
    "ai_server_llm_fallbacks",
    "Successful uses of a non-primary language-model provider.",
    ("provider", "profile"),
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


def _usage_metadata(response) -> dict | None:
    """Read message usage, falling back to provider-reported token totals."""
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                return usage
    usage = (response.llm_output or {}).get("token_usage")
    return usage if isinstance(usage, dict) else None


def _token_counts(usage: dict) -> tuple[int, int] | None:
    """Return non-negative integer input and output token counts."""
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        return None
    return input_tokens, output_tokens


class LLMUsageCallback(BaseCallbackHandler):
    """Record bounded token, request, fallback, and OpenAI cost metrics."""

    def __init__(
        self,
        *,
        provider: str,
        profile: str,
        is_fallback: bool,
        input_cost_per_million_usd: float | None = None,
        output_cost_per_million_usd: float | None = None,
        cached_input_cost_per_million_usd: float | None = None,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.is_fallback = is_fallback
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd
        self.cached_input_cost_per_million_usd = cached_input_cost_per_million_usd

    def on_llm_end(self, response, **kwargs) -> None:
        """Record completed calls, valid token usage, fallbacks, and cost."""
        LLM_REQUESTS.labels(self.provider, self.profile, "success").inc()
        if self.is_fallback:
            LLM_FALLBACKS.labels(self.provider, self.profile).inc()

        usage = _usage_metadata(response)
        if not usage:
            return
        token_counts = _token_counts(usage)
        if token_counts is None:
            return
        input_tokens, output_tokens = token_counts
        LLM_TOKENS.labels(self.provider, self.profile, "input").inc(input_tokens)
        LLM_TOKENS.labels(self.provider, self.profile, "output").inc(output_tokens)
        self._record_openai_cost(usage, input_tokens, output_tokens)

    def _record_openai_cost(
        self,
        usage: dict,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Estimate cost from configured rates and reported cache token details."""
        if self.input_cost_per_million_usd is None or self.output_cost_per_million_usd is None:
            return
        input_details = usage.get("input_token_details") or {}
        cached_input_tokens = min(
            input_tokens,
            input_details.get("cache_read", input_details.get("cached_tokens", 0)) or 0,
        )
        cache_write_tokens = min(
            input_tokens - cached_input_tokens,
            input_details.get("cache_creation", 0) or 0,
        )
        uncached_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
        cached_rate = self.cached_input_cost_per_million_usd or self.input_cost_per_million_usd
        cost = (
            uncached_input_tokens * self.input_cost_per_million_usd
            + cached_input_tokens * cached_rate
            + cache_write_tokens * self.input_cost_per_million_usd * 1.25
            + output_tokens * self.output_cost_per_million_usd
        ) / 1_000_000
        if cost:
            LLM_COST_USD.labels(self.provider, self.profile).inc(cost)

    def on_llm_error(self, error, **kwargs) -> None:
        LLM_REQUESTS.labels(self.provider, self.profile, "error").inc()


def observe_http_request(
    method: str,
    route: str,
    status: int | str,
    duration_seconds: float,
) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_REQUEST_DURATION.labels(method, route).observe(duration_seconds)


def observe_agent_execution(agent: str, duration_seconds: float) -> None:
    """Record the execution duration for one agent node."""
    AGENT_EXECUTION_DURATION.labels(agent).observe(duration_seconds)


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
