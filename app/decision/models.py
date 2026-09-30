from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DecisionInput(BaseModel):
    """Small, minimized state sent to a decision provider."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4_000)
    available_agents: list[str] = Field(min_length=2, max_length=32)
    agent_tools: dict[str, list[str]] = Field(max_length=32)
    context: dict[str, bool | str] = Field(default_factory=dict, max_length=16)

    @property
    def available_tools(self) -> list[str]:
        """Return the distinct tools available across all candidate agents."""
        return sorted({tool for tools in self.agent_tools.values() for tool in tools})

    def to_provider_state(self) -> dict[str, object]:
        """Return only the bounded information required to choose a route."""
        return {
            "message": self.message,
            "available_agents": self.available_agents,
            "available_tools": self.available_tools,
            "agent_tools": self.agent_tools,
            "context": self.context,
        }


class MidasDecision(BaseModel):
    """Validated route, tool and execution-strategy proposal."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1, max_length=64)
    tools: list[str] = Field(default_factory=list, max_length=16)
    needs_mcp: bool
    needs_analytics: bool
    confidence: float = Field(ge=0, le=1)

    @field_validator("tools")
    @classmethod
    def unique_tools(cls, value: list[str]) -> list[str]:
        """Reject duplicate tool choices returned by the decision provider."""
        if len(value) != len(set(value)):
            raise ValueError("decision tools must be unique")
        return value


class ProviderResult(BaseModel):
    """Decision plus bounded provider usage metadata."""

    decision: MidasDecision
    model: str = Field(min_length=1, max_length=100)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DecisionOutcome(BaseModel):
    """Result of decision evaluation, including an explicit fallback reason."""

    decision: MidasDecision
    fallback_reason: str | None = None
    model: str = ""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def accepted(self) -> bool:
        """Report whether this decision can replace legacy routing."""
        return self.fallback_reason is None


DecisionStatus = Literal[
    "success",
    "disabled",
    "low_confidence",
    "timeout",
    "provider_error",
    "invalid_schema",
    "invalid_decision",
    "call_limit",
]
