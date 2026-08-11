from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage

from app.schemas.chat import ChatRequest, ChatResponse


async def invoke_graph(graph, payload: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": payload.thread_id}}
    snapshot = await graph.aget_state(config)
    owner_id = snapshot.values.get("user_id")
    if owner_id and owner_id != payload.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta conversa pertence a outro usuario.",
        )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=payload.message)],
            "user_id": payload.user_id,
            "route": "",
            "agents": [],
        },
        config=config,
    )
    message = result["messages"][-1].content
    return ChatResponse(thread_id=payload.thread_id, message=message, agents=result["agents"])
