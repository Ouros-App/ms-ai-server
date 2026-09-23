import asyncio
import logging
from contextlib import nullcontext
from time import perf_counter

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage

from app.agents.diagnostics import specialist_results_summary
from app.agents.guardrails import guard_input
from app.agents.mcp import forward_mcp_access_token
from app.core.config import settings
from app.core.metrics import (
    observe_chat_duration,
    observe_chat_result,
    observe_chat_routing,
)
from app.debug_ui.trace import capture_debug_trace, trace_event
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


async def _invoke_graph(
    graph,
    payload: ChatRequest,
    principal_id: str,
    thread_ownership=None,
    principal_token: str | None = None,
    *,
    debug: bool = False,
) -> tuple[ChatResponse, dict]:
    started_at = perf_counter()
    config = {"configurable": {"thread_id": payload.thread_id}}
    trace_context = capture_debug_trace() if debug else nullcontext([])

    with trace_context as trace:
        trace_event(
            "request.started",
            thread_id=payload.thread_id,
            authenticated=True,
        )

        if thread_ownership is not None and not await thread_ownership.claim(
            payload.thread_id,
            principal_id,
        ):
            trace_event("thread.denied", reason="owned_by_another_user")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Esta conversa pertence a outro usuario.",
            )

        snapshot = await graph.aget_state(config)
        owner_id = snapshot.values.get("user_id")
        has_history = bool(snapshot.values.get("messages"))
        trace_event(
            "thread.loaded",
            has_history=has_history,
            owner_bound=bool(owner_id),
        )
        if thread_ownership is None and owner_id and owner_id != principal_id:
            trace_event("thread.denied", reason="snapshot_owner_mismatch")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Esta conversa pertence a outro usuario.",
            )

        input_guardrail = await guard_input(
            payload.message,
            has_history=has_history,
            user_id=principal_id,
        )
        guardrail_state = input_guardrail.as_state()
        trace_event(
            "guardrail.input",
            allowed=input_guardrail.allowed,
            category=input_guardrail.category,
            state=guardrail_state,
        )
        if not input_guardrail.allowed:
            observe_chat_result("blocked", ["guardrail"], [])
            observe_chat_duration("blocked", perf_counter() - started_at)
            logger.warning(
                "chat_blocked thread_id=%s category=%s",
                payload.thread_id,
                input_guardrail.category,
            )
            trace_event("request.blocked", category=input_guardrail.category)
            response = ChatResponse(
                thread_id=payload.thread_id,
                message=input_guardrail.message,
                agents=["guardrail"],
                tools=[],
            )
            return response, {
                "routes": ["guardrail"],
                "specialist_results": [],
                "guardrail": guardrail_state,
                "trace": trace,
                "duration_ms": round((perf_counter() - started_at) * 1000, 1),
            }

        safe_payload = payload.model_copy(
            update={"message": input_guardrail.sanitized_text}
        )

        try:
            trace_event("graph.started")
            async with asyncio.timeout(settings.llm_total_timeout_seconds):
                with forward_mcp_access_token(principal_token):
                    result = await graph.ainvoke(
                        {
                            "messages": [HumanMessage(content=safe_payload.message)],
                            "user_id": principal_id,
                            "route": "",
                            "routes": [],
                            "agents": [],
                            "tools": [],
                            "specialist_results": [],
                            "input_guardrail": guardrail_state,
                        },
                        config=config,
                    )
        except TimeoutError as error:
            observe_chat_result("timeout", ["graph"], [])
            observe_chat_duration("timeout", perf_counter() - started_at)
            trace_event(
                "graph.timeout",
                timeout_seconds=settings.llm_total_timeout_seconds,
            )
            logger.warning(
                "chat_provider_timeout thread_id=%s timeout_seconds=%s",
                payload.thread_id,
                settings.llm_total_timeout_seconds,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="O provedor de IA demorou para responder. Tente novamente.",
            ) from error
        except Exception:
            observe_chat_result("error", ["graph"], [])
            observe_chat_duration("error", perf_counter() - started_at)
            raise

        message = result["messages"][-1].content
        tools = result.get("tools", [])
        agents = result.get("agents", [])
        routes = result.get("routes", [])
        route_source = result.get("route_source")
        specialist_results = result.get("specialist_results", [])
        debug_specialist_results = specialist_results_summary(specialist_results)
        pending_routes = result.get("pending_routes", [])
        pending_missing_data = result.get("pending_missing_data", [])
        pending_by_route = result.get("pending_by_route", {})
        pending_personal_routes = result.get("pending_personal_routes", [])
        duration_ms = round((perf_counter() - started_at) * 1000, 1)

        observe_chat_result("success", agents, tools)
        observe_chat_duration("success", duration_ms / 1000)
        observe_chat_routing(routes, route_source, pending_routes)
        trace_event(
            "graph.completed",
            routes=routes,
            route_source=route_source,
            agents=agents,
            tools=tools,
            specialist_results=debug_specialist_results,
            pending_routes=pending_routes,
            pending_missing_data=pending_missing_data,
            pending_by_route=pending_by_route,
            pending_personal_routes=pending_personal_routes,
            duration_ms=duration_ms,
        )
        logger.info(
            "chat_completed thread_id=%s agents=%s tools=%s duration_ms=%.1f",
            payload.thread_id,
            agents,
            tools,
            duration_ms,
        )
        response = ChatResponse(
            thread_id=payload.thread_id,
            message=message,
            agents=agents,
            tools=tools,
        )
        return response, {
            "routes": routes,
            "route_source": route_source,
            "specialist_results": debug_specialist_results,
            "pending_routes": pending_routes,
            "pending_missing_data": pending_missing_data,
            "pending_by_route": pending_by_route,
            "pending_personal_routes": pending_personal_routes,
            "guardrail": guardrail_state,
            "trace": trace,
            "duration_ms": duration_ms,
        }


async def invoke_graph(
    graph,
    payload: ChatRequest,
    principal_id: str,
    thread_ownership=None,
    principal_token: str | None = None,
) -> ChatResponse:
    """Validate ownership, execute the graph and return the product response."""

    response, _diagnostics = await _invoke_graph(
        graph,
        payload,
        principal_id,
        thread_ownership,
        principal_token,
        debug=False,
    )
    return response


async def invoke_graph_debug(
    graph,
    payload: ChatRequest,
    principal_id: str,
    thread_ownership=None,
    principal_token: str | None = None,
) -> tuple[ChatResponse, dict]:
    """Run the same product path while collecting request-local diagnostics."""

    return await _invoke_graph(
        graph,
        payload,
        principal_id,
        thread_ownership,
        principal_token,
        debug=True,
    )
