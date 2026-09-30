import asyncio
import logging
from time import perf_counter

from app.core.config import settings
from app.core.metrics import observe_jev_request, observe_jev_router_agreement
from app.debug_ui.trace import trace_event
from app.decision.fallback import DeterministicFallbackProvider
from app.decision.jev_provider import (
    DecisionProviderError,
    DecisionTimeoutError,
    InvalidDecisionError,
    JevDecisionProvider,
)
from app.decision.models import DecisionInput, DecisionOutcome, ProviderResult
from app.decision.provider import DecisionProvider

logger = logging.getLogger(__name__)


class DecisionService:
    """Validate provider decisions and fall back without affecting Midas safety."""

    def __init__(
        self,
        provider: DecisionProvider,
        fallback_provider: DeterministicFallbackProvider,
        *,
        enabled: bool,
        min_confidence: float,
        max_calls_per_request: int,
        input_cost_per_million_usd: float,
        output_cost_per_million_usd: float,
    ) -> None:
        """Configure provider validation, fallback, cost, and call limits."""
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.max_calls_per_request = max_calls_per_request
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd
        self._background_tasks: set[asyncio.Task] = set()

    @classmethod
    def from_settings(cls) -> "DecisionService":
        """Create the decision service from process configuration."""
        return cls(
            provider=JevDecisionProvider.from_settings(),
            fallback_provider=DeterministicFallbackProvider(),
            enabled=settings.jev_enabled,
            min_confidence=settings.jev_min_confidence,
            max_calls_per_request=settings.jev_max_calls_per_request,
            input_cost_per_million_usd=settings.jev_input_cost_per_million_usd,
            output_cost_per_million_usd=settings.jev_output_cost_per_million_usd,
        )

    @property
    def ready(self) -> bool:
        """Return whether shadow requests can reach a configured provider."""
        return self.enabled and bool(getattr(self.provider, "api_key", None))

    async def decide(
        self,
        state: DecisionInput,
        *,
        call_number: int = 1,
    ) -> DecisionOutcome:
        """Obtain a decision or return a safe fallback with telemetry."""
        started_at = perf_counter()
        result: ProviderResult | None = None
        if not self.enabled:
            return self._fallback(state, "disabled", started_at)
        if call_number > self.max_calls_per_request:
            return self._fallback(state, "call_limit", started_at)

        trace_event("jev.decision.started", decision_type="combined")
        try:
            result = await self.provider.decide(state)
            self._validate_decision(result, state)
            if result.decision.confidence < self.min_confidence:
                return self._fallback(
                    state,
                    "low_confidence",
                    started_at,
                    result=result,
                )
            duration = perf_counter() - started_at
            self._observe("success", duration, result=result)
            outcome = DecisionOutcome(
                decision=result.decision,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            self._log_completed(outcome, duration)
            return outcome
        except DecisionTimeoutError as error:
            reason = error.reason
        except InvalidDecisionError as error:
            reason = error.reason
        except DecisionProviderError as error:
            reason = error.reason
        except Exception:
            logger.exception("jev_decision_unexpected_error")
            reason = "provider_error"
        return self._fallback(
            state,
            reason,
            started_at,
            result=result,
        )

    @staticmethod
    def _validate_decision(result: ProviderResult, state: DecisionInput) -> None:
        """Enforce available routes, route tools, and backend capabilities."""
        decision = result.decision
        if decision.agent not in state.available_agents:
            raise InvalidDecisionError("Jev selected an unavailable Midas route")
        allowed_tools = set(state.agent_tools.get(decision.agent, []))
        if not set(decision.tools).issubset(allowed_tools):
            raise InvalidDecisionError("Jev selected a tool outside the route allowlist")
        if decision.needs_mcp != bool(decision.tools):
            raise InvalidDecisionError("Jev MCP strategy conflicts with tool selection")
        if decision.needs_analytics and not state.context.get("has_analytics", False):
            raise InvalidDecisionError("Jev selected an unavailable analytics backend")

    def _fallback(
        self,
        state: DecisionInput,
        reason: str,
        started_at: float,
        *,
        result: ProviderResult | None = None,
    ) -> DecisionOutcome:
        """Create a fallback outcome and record why Jev was not applied."""
        fallback_result = self.fallback_provider.decide(state)
        duration = perf_counter() - started_at
        tokens_in = result.input_tokens if result else 0
        tokens_out = result.output_tokens if result else 0
        self._observe(
            reason,
            duration,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            fallback_reason=reason,
            result=result,
        )
        outcome = DecisionOutcome(
            decision=fallback_result.decision,
            fallback_reason=reason,
            model=result.model if result else fallback_result.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )
        self._log_completed(outcome, duration)
        trace_event(
            "jev.decision.fallback",
            reason=reason,
            duration_ms=round(duration * 1000, 1),
        )
        return outcome

    def _observe(
        self,
        status: str,
        duration: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        fallback_reason: str | None = None,
        result: ProviderResult | None = None,
    ) -> None:
        """Record bounded token, latency, and estimated cost metrics."""
        if result is not None:
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
        cost = (
            input_tokens * self.input_cost_per_million_usd
            + output_tokens * self.output_cost_per_million_usd
        ) / 1_000_000
        observe_jev_request(
            status,
            duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _log_completed(outcome: DecisionOutcome, duration: float) -> None:
        """Write structured decision details without request contents."""
        logger.info(
            "jev.decision.completed agent=%s tools=%s confidence=%.3f "
            "duration_ms=%.1f fallback=%s reason=%s input_tokens=%d "
            "output_tokens=%d",
            outcome.decision.agent,
            outcome.decision.tools,
            outcome.decision.confidence,
            duration * 1000,
            not outcome.accepted,
            outcome.fallback_reason or "none",
            outcome.input_tokens,
            outcome.output_tokens,
        )
        trace_event(
            "jev.decision.completed",
            agent=outcome.decision.agent,
            tools=outcome.decision.tools,
            confidence=outcome.decision.confidence,
            duration_ms=round(duration * 1000, 1),
            fallback=not outcome.accepted,
            reason=outcome.fallback_reason,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )

    def schedule_shadow(self, state: DecisionInput) -> asyncio.Task | None:
        """Run a bounded shadow request without delaying the user's response."""
        if not self.ready or self.max_calls_per_request < 1:
            return None
        task = asyncio.create_task(self.decide(state), name="jev-shadow-decision")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @staticmethod
    def record_shadow_agreement(
        task: asyncio.Task,
        active_agent: str,
    ) -> None:
        """Compare a completed shadow result with the active router safely."""
        try:
            outcome: DecisionOutcome = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("jev_shadow_result_failed")
            return
        if not outcome.accepted:
            return
        agreement = outcome.decision.agent == active_agent
        observe_jev_router_agreement(agreement)
        logger.info(
            "jev.shadow.comparison active_agent=%s jev_agent=%s agreement=%s",
            active_agent,
            outcome.decision.agent,
            agreement,
        )
        trace_event(
            "jev.shadow.comparison",
            active_agent=active_agent,
            jev_agent=outcome.decision.agent,
            agreement=agreement,
        )
