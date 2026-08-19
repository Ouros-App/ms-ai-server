from typing import Literal

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]
    next_cursor: str | None = None
