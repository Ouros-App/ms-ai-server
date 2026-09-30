from app.decision.models import DecisionInput, MidasDecision, ProviderResult


class DeterministicFallbackProvider:
    """Provide a safe sentinel that hands control to the existing router."""

    def decide(self, state: DecisionInput) -> ProviderResult:
        fallback_agent = (
            "default" if "default" in state.available_agents else state.available_agents[-1]
        )
        return ProviderResult(
            decision=MidasDecision(
                agent=fallback_agent,
                tools=[],
                needs_mcp=False,
                needs_analytics=False,
                confidence=0,
            ),
            model="deterministic-fallback",
            input_tokens=0,
            output_tokens=0,
        )
