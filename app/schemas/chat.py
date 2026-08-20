from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8_000)
    thread_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)


class ChatResponse(BaseModel):
    thread_id: str
    message: str
    agents: list[str]
    tools: list[str]
