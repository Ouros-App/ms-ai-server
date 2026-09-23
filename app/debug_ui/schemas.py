from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class DebugLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=255)
    password: SecretStr = Field(min_length=1)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.count("@") != 1 or " " in normalized:
            raise ValueError("invalid email")
        return normalized


class DebugSessionResponse(BaseModel):
    authenticated: bool = True
    user_id: str
    account_type: str


class DebugChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
    )


class DebugSpecialistResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    status: str
    fact_count: int = Field(default=0, ge=0)
    recommendation_count: int = Field(default=0, ge=0)
    missing_data: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)


class DebugChatResponse(BaseModel):
    thread_id: str
    message: str
    agents: list[str]
    tools: list[str]
    routes: list[str]
    route_source: str | None = None
    specialist_results: list[DebugSpecialistResult] = Field(default_factory=list)
    pending_routes: list[str] = Field(default_factory=list)
    pending_missing_data: list[str] = Field(default_factory=list)
    pending_by_route: dict[str, list[str]] = Field(default_factory=dict)
    pending_personal_routes: list[str] = Field(default_factory=list)
    guardrail: dict
    trace: list[dict]
    duration_ms: float
