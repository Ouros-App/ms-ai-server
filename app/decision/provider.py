from typing import Protocol

from app.decision.models import DecisionInput, ProviderResult


class DecisionProvider(Protocol):
    """Provider boundary for structured Midas decisions."""

    async def decide(self, state: DecisionInput) -> ProviderResult:
        """Evaluate one minimized state and return typed decisions."""
        ...
