from fastapi import HTTPException, status

from app.schemas.history import HistoryMessage, HistoryResponse


async def get_thread_history(
    checkpointer,
    thread_id: str,
    user_id: str,
    limit: int,
    before: str | None,
) -> HistoryResponse:
    """Busca uma pagina do historico persistido de uma thread pertencente ao usuario."""
    checkpoint = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id}},
    )
    if checkpoint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversa nao encontrada.",
        )

    values = checkpoint.checkpoint.get("channel_values", {})
    if values.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta conversa pertence a outro usuario.",
        )

    messages = _visible_messages(values.get("messages", []))
    end = len(messages) if before is None else _parse_cursor(before, len(messages))
    start = max(0, end - limit)
    return HistoryResponse(
        thread_id=thread_id,
        messages=messages[start:end],
        next_cursor=str(start) if start else None,
    )


def _parse_cursor(cursor: str, message_count: int) -> int:
    try:
        position = int(cursor)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="O cursor before e invalido.",
        ) from error
    if position < 0 or position > message_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="O cursor before e invalido.",
        )
    return position


def _visible_messages(messages: list) -> list[HistoryMessage]:
    roles = {"human": "user", "ai": "assistant"}
    return [
        HistoryMessage(role=roles[message.type], content=message.content)
        for message in messages
        if message.type in roles and isinstance(message.content, str) and message.content
    ]
