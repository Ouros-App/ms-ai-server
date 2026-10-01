from fastapi import HTTPException, status

from app.schemas.history import HistoryMessage, HistoryResponse


async def get_thread_history(
    checkpointer,
    thread_id: str,
    user_id: str,
    limit: int,
    before: str | None,
    visualization_store=None,
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

    checkpoint_messages = [
        message
        for message in values.get("messages", [])
        if message.type in {"human", "ai"}
        and isinstance(message.content, str)
        and message.content
    ]
    end = (
        len(checkpoint_messages)
        if before is None
        else _parse_cursor(before, len(checkpoint_messages))
    )
    start = max(0, end - limit)
    checkpoint_page = checkpoint_messages[start:end]
    visualization_ids = [
        message.additional_kwargs.get("visualization_id")
        for message in checkpoint_page
        if message.type == "ai"
        and isinstance(message.additional_kwargs, dict)
        and isinstance(message.additional_kwargs.get("visualization_id"), str)
    ]
    visualizations_by_id = (
        await visualization_store.get_many(thread_id, user_id, visualization_ids)
        if visualization_store is not None and visualization_ids
        else {}
    )
    messages = _visible_messages(checkpoint_page, visualizations_by_id)
    return HistoryResponse(
        thread_id=thread_id,
        messages=messages,
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


def _visible_messages(
    messages: list,
    visualizations_by_id: dict[str, list[dict]] | None = None,
) -> list[HistoryMessage]:
    roles = {"human": "user", "ai": "assistant"}
    visualizations_by_id = visualizations_by_id or {}
    visible = []
    for message in messages:
        if (
            message.type not in roles
            or not isinstance(message.content, str)
            or not message.content
        ):
            continue
        additional_kwargs = getattr(message, "additional_kwargs", {})
        visualization_id = (
            additional_kwargs.get("visualization_id")
            if isinstance(additional_kwargs, dict)
            else None
        )
        visible.append(
            HistoryMessage(
                role=roles[message.type],
                content=message.content,
                visualizations=visualizations_by_id.get(visualization_id, []),
            )
        )
    return visible
