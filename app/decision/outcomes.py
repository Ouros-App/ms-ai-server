import asyncio
import logging
from time import perf_counter

from app.agents.guardrails import anonymize_text, guard_output
from app.core.config import settings
from app.core.metrics import observe_jev_outcome
from app.decision.jev_provider import JevDecisionProvider

logger = logging.getLogger(__name__)


class JevOutcomeService:
    """Classify completed chats asynchronously for estimated outcome metrics."""

    def __init__(self, provider: JevDecisionProvider) -> None:
        """Retain the provider and scheduled task references."""
        self.provider = provider
        self._tasks: set[asyncio.Task] = set()

    @classmethod
    def from_settings(cls) -> "JevOutcomeService":
        """Build the classifier from the active Jev configuration."""
        return cls(JevDecisionProvider.from_settings())

    def schedule(
        self,
        request: str,
        response: str,
        *,
        decision_calls: int,
    ) -> None:
        """Queue a PII-scrubbed classification within the remaining call budget."""
        if not settings.jev_enabled or not self.provider.api_key:
            return
        calls_used = decision_calls + int(settings.jev_shadow_mode)
        if calls_used >= settings.jev_max_calls_per_request:
            return

        safe_request, _ = anonymize_text(request[:4_000])
        safe_response, _ = anonymize_text(guard_output(response)[:4_000])
        task = asyncio.create_task(self._classify(safe_request, safe_response))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _classify(self, request: str, response: str) -> None:
        """Classify one chat turn and record bounded outcome and cost metrics."""
        started_at = perf_counter()
        try:
            result = await self.provider.classify_outcome(request, response)
        except Exception as error:  # noqa: BLE001 - isolate background failures from chat
            logger.warning(
                "jev.outcome_classification.failed error_type=%s",
                type(error).__name__,
            )
            observe_jev_outcome(
                "unavailable",
                "error",
                perf_counter() - started_at,
            )
            return

        outcome = (
            result.outcome
            if result.confidence >= settings.jev_min_confidence
            else "uncertain"
        )
        cost = (
            result.input_tokens * settings.jev_input_cost_per_million_usd
            + result.output_tokens * settings.jev_output_cost_per_million_usd
        ) / 1_000_000
        observe_jev_outcome(
            outcome,
            "success",
            perf_counter() - started_at,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
        )
        logger.info(
            "jev.outcome_classification.completed outcome=%s confidence=%.3f model=%s",
            outcome,
            result.confidence,
            result.model,
        )


jev_outcome_service = JevOutcomeService.from_settings()
