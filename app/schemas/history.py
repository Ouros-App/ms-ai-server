from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatVisualization


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    visualizations: list[ChatVisualization] = Field(default_factory=list, max_length=1)


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]
    next_cursor: str | None = None
