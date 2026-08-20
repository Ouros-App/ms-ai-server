import logging
from time import perf_counter

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage

from app.agents.guardrails import guard_input
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


async def invoke_graph(
    graph,
    payload: ChatRequest,
    principal_id: str,
    thread_ownership=None,
) -> ChatResponse:
    """Valida a posse da thread, executa o grafo e formata a resposta."""
    started_at = perf_counter()
    config = {"configurable": {"thread_id": payload.thread_id}}

    if thread_ownership is not None and not await thread_ownership.claim(
        payload.thread_id,
        principal_id,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta conversa pertence a outro usuario.",
        )

    snapshot = await graph.aget_state(config)
    owner_id = snapshot.values.get("user_id")
    if thread_ownership is None and owner_id and owner_id != principal_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta conversa pertence a outro usuario.",
        )

    input_guardrail = await guard_input(
        payload.message,
        has_history=bool(snapshot.values.get("messages")),
    )
    if not input_guardrail.allowed:
        logger.warning(
            "chat_blocked thread_id=%s category=%s",
            payload.thread_id,
            input_guardrail.category,
        )
        return ChatResponse(
            thread_id=payload.thread_id,
            message=input_guardrail.message,
            agents=["guardrail"],
            tools=[],
        )

    safe_payload = payload.model_copy(update={"message": input_guardrail.sanitized_text})

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=safe_payload.message)],
            "user_id": principal_id,
            "route": "",
            "agents": [],
            "tools": [],
            "input_guardrail": input_guardrail.as_state(),
        },
        config=config,
    )
    message = result["messages"][-1].content
    tools = result.get("tools", [])
    logger.info(
        "chat_completed thread_id=%s agents=%s tools=%s duration_ms=%.1f",
        payload.thread_id,
        result["agents"],
        tools,
        (perf_counter() - started_at) * 1000,
    )
    return ChatResponse(
        thread_id=payload.thread_id,
        message=message,
        agents=result["agents"],
        tools=tools,
    )
