import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Gauge, Histogram

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
CHAT_TURN_DURATION = Histogram(
    "ai_server_chat_turn_duration_seconds",
    "Duracao total dos turnos do chat em segundos.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 30, 60),
)
CHAT_TURN_OUTCOMES = Counter(
    "ai_server_chat_turns",
    "Turnos de chat por resultado.",
    ("outcome",),
)
CHAT_TURNS_IN_PROGRESS = Gauge(
    "ai_server_chat_turns_in_progress",
    "Quantidade de turnos de chat em andamento.",
)
LLM_CALLS = Counter(
    "ai_server_llm_calls",
    "Chamadas a modelos por provedor, modelo e resultado.",
    ("provider", "model", "node", "outcome"),
)
LLM_DURATION = Histogram(
    "ai_server_llm_call_duration_seconds",
    "Duracao das chamadas a modelos em segundos.",
    ("provider", "model", "node"),
)
LLM_INPUT_TOKENS = Counter(
    "ai_server_llm_input_tokens",
    "Tokens de entrada reportados pelos modelos.",
    ("provider", "model", "node"),
)
LLM_OUTPUT_TOKENS = Counter(
    "ai_server_llm_output_tokens",
    "Tokens de saida reportados pelos modelos.",
    ("provider", "model", "node"),
)
LLM_ESTIMATED_COST_USD = Counter(
    "ai_server_llm_estimated_cost_usd",
    "Custo estimado em USD com as tarifas configuradas.",
    ("provider", "model", "node"),
)
LLM_FALLBACKS = Counter(
    "ai_server_llm_fallbacks",
    "Chamadas primarias que falharam antes de outro modelo responder.",
    ("provider", "model", "node"),
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

# USD per million tokens, from https://console.groq.com/docs/models (2026-10-09).
# NVIDIA API Catalog has no published per-token rate; NIM production licensing
# is GPU-based, so NVIDIA calls intentionally have no token cost estimate.
LLM_PRICE_PER_MILLION_USD = {
    "groq_gpt_oss_20b": (0.075, 0.30),
    "groq_gpt_oss_120b": (0.15, 0.60),
}

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


def observe_chat_turn(duration_seconds: float, outcome: str) -> None:
    CHAT_TURN_DURATION.observe(duration_seconds)
    CHAT_TURN_OUTCOMES.labels(outcome).inc()


def _model_identity(serialized: dict[str, Any] | None) -> tuple[str, str]:
    kwargs = (serialized or {}).get("kwargs") or {}
    invocation_params = kwargs.get("invocation_params") or {}
    model = str(
        kwargs.get("model")
        or kwargs.get("model_name")
        or invocation_params.get("model")
        or invocation_params.get("model_name")
        or ""
    ).lower()
    model_id = str((serialized or {}).get("id", "")).lower()
    if "groq" in model_id or model.startswith("openai/"):
        provider = "groq"
    elif "nvidia" in model or "nim" in str(
        kwargs.get("openai_api_base")
        or kwargs.get("base_url")
        or invocation_params.get("base_url")
        or ""
    ).lower():
        provider = "nvidia"
    else:
        provider = "other"
    if "gpt-oss-20b" in model:
        model_name = "groq_gpt_oss_20b"
    elif "gpt-oss-120b" in model:
        model_name = "groq_gpt_oss_120b"
    elif "nemotron-3-nano-30b-a3b" in model:
        model_name = "nvidia_nemotron_3_nano_30b"
    elif "nemotron-3-super-120b-a12b" in model:
        model_name = "nvidia_nemotron_3_super_120b"
    else:
        model_name = "other"
    return provider, model_name


_LLM_NODES = {
    "router",
    "collect_specialist_results",
    "default",
    "guardrail",
    "faq",
    "support",
    "sustainability",
    "ranking",
}


def _safe_llm_node(
    metadata: dict[str, Any] | None,
    override: str | None = None,
) -> str:
    if override in _LLM_NODES:
        return override
    node = (metadata or {}).get("langgraph_node")
    return node if isinstance(node, str) and node in _LLM_NODES else "unknown"


def _token_usage(response: LLMResult) -> tuple[int, int]:
    usage = response.llm_output or {}
    usage = usage.get("token_usage") or usage.get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if not prompt and not completion:
        for generations in response.generations:
            for generation in generations:
                message_usage = getattr(generation.message, "usage_metadata", None) or {}
                prompt += message_usage.get("input_tokens", 0)
                completion += message_usage.get("output_tokens", 0)
    return int(prompt or 0), int(completion or 0)


class LLMMetricsCallback(BaseCallbackHandler):
    """Record bounded model-call metrics without retaining prompts or outputs."""

    def __init__(self, node: str | None = None) -> None:
        self._lock = threading.Lock()
        self._runs: dict[UUID, tuple[float, str, str, str, UUID | None]] = {}
        self._failed: dict[UUID | None, list[tuple[str, str]]] = {}
        self._node = node

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        provider, model = _model_identity(serialized)
        node = _safe_llm_node(kwargs.get("metadata"), self._node)
        parent_run_id = kwargs.get("parent_run_id")
        with self._lock:
            self._runs[run_id] = (
                time.perf_counter(), provider, model, node, parent_run_id
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        provider, model = _model_identity(serialized)
        node = _safe_llm_node(kwargs.get("metadata"), self._node)
        parent_run_id = kwargs.get("parent_run_id")
        with self._lock:
            self._runs[run_id] = (
                time.perf_counter(), provider, model, node, parent_run_id
            )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        started, provider, model, node, parent_run_id = run
        LLM_CALLS.labels(provider, model, node, "error").inc()
        LLM_DURATION.labels(provider, model, node).observe(
            time.perf_counter() - started
        )
        with self._lock:
            self._failed.setdefault(parent_run_id, []).append((provider, model))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        started, provider, model, node, parent_run_id = run
        duration = time.perf_counter() - started
        input_tokens, output_tokens = _token_usage(response)
        LLM_CALLS.labels(provider, model, node, "success").inc()
        LLM_DURATION.labels(provider, model, node).observe(duration)
        LLM_INPUT_TOKENS.labels(provider, model, node).inc(input_tokens)
        LLM_OUTPUT_TOKENS.labels(provider, model, node).inc(output_tokens)
        input_rate, output_rate = LLM_PRICE_PER_MILLION_USD.get(model, (0, 0))
        estimated_cost = (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000
        if estimated_cost:
            LLM_ESTIMATED_COST_USD.labels(provider, model, node).inc(estimated_cost)
        with self._lock:
            failed = self._failed.pop(parent_run_id, [])
            for failed_provider, failed_model in failed:
                LLM_FALLBACKS.labels(
                    failed_provider, failed_model, node
                ).inc()


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
