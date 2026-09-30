from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8_000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=128
    )


class ChatVisualizationChart(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    render_as: str = Field(min_length=1, max_length=32)
    html: str = Field(min_length=1, max_length=1_500_000)


class ChatVisualization(BaseModel):
    type: Literal["ouros_dashboard"]
    title: str = Field(min_length=1, max_length=120)
    charts: list[ChatVisualizationChart] = Field(min_length=1, max_length=4)


class ChatResponse(BaseModel):
    thread_id: str
    message: str
    agents: list[str]
    tools: list[str]
    visualizations: list[ChatVisualization] = Field(default_factory=list, max_length=1)
