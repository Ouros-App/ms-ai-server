import logging
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.decision.models import DecisionInput, MidasDecision, ProviderResult

logger = logging.getLogger(__name__)


class DecisionProviderError(RuntimeError):
    """Base error for failures that should trigger the existing router."""

    reason = "provider_error"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        if reason is not None:
            self.reason = reason


class DecisionTimeoutError(DecisionProviderError):
    """The decision provider exceeded its request deadline."""

    reason = "timeout"


class InvalidDecisionError(DecisionProviderError):
    """The decision provider returned a missing or invalid answer."""

    reason = "invalid_schema"


_AGENT_DESCRIPTIONS = {
    "faq": "Regras e funcionalidades confirmadas do produto e do aplicativo.",
    "sustainability": "Consumo de água e energia e sustentabilidade.",
    "ranking": "Indicadores, histórico, resultados e ranking.",
    "visualization": "Criar um gráfico ou painel temporário para o chat.",
    "support": "Problemas de uso, acesso, sincronização e suporte técnico.",
    "default": "Resposta geral sintetizada a partir das demais informações.",
}
_TOOL_DESCRIPTIONS = {
    "search_knowledge": "Consultar conteúdo oficial sobre o produto.",
    "get_user_context": "Ler contexto legível da conta autenticada.",
    "get_consumption_summary": "Consultar consumo autenticado por período.",
    "create_custom_dashboard": "Criar gráfico usando o catálogo permitido.",
}


class JevDecisionProvider:
    """Call TypeSafe System One and validate its structured response."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Store provider credentials and bounded request settings."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @classmethod
    def from_settings(cls) -> "JevDecisionProvider":
        """Build the TypeSafe client from the application settings."""
        key = settings.jev_api_key.get_secret_value() if settings.jev_api_key else None
        return cls(
            api_key=key,
            base_url=settings.jev_base_url,
            model=settings.jev_model,
            timeout_seconds=settings.jev_timeout_ms / 1_000,
        )

    async def decide(self, state: DecisionInput) -> ProviderResult:
        """Request and validate one structured decision from System One."""
        if not self.api_key:
            raise DecisionProviderError(
                "JEV_API_KEY is not configured",
                reason="api_key_missing",
            )

        questions = self._questions(state)
        payload = {
            "model": self.model,
            "state": state.to_provider_state(),
            "questions": questions,
        }
        try:
            provider_host = urlsplit(self.base_url).hostname or "unknown"
            logger.info(
                "jev.provider_request.started model=%s host=%s timeout_ms=%d",
                self.model,
                provider_host,
                round(self.timeout_seconds * 1000),
            )
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/v1/systemone",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
        except httpx.TimeoutException as error:
            logger.warning(
                "jev.provider_request.failed reason=timeout model=%s timeout_ms=%d",
                self.model,
                round(self.timeout_seconds * 1000),
            )
            raise DecisionTimeoutError("Jev decision timed out") from error
        except httpx.HTTPError as error:
            logger.warning(
                "jev.provider_request.failed reason=transport_error model=%s error_type=%s",
                self.model,
                type(error).__name__,
            )
            raise DecisionProviderError("Jev request failed") from error

        logger.info(
            "jev.provider_request.completed model=%s status=%d",
            self.model,
            response.status_code,
        )

        if response.status_code == 429 or response.status_code >= 500:
            raise DecisionProviderError(
                f"Jev service returned HTTP {response.status_code}"
            )
        if response.status_code < 200 or response.status_code >= 300:
            raise DecisionProviderError(
                f"Jev request was rejected with HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise InvalidDecisionError("Jev response was not valid JSON") from error

        return self._parse_response(body, state)

    @staticmethod
    def _questions(state: DecisionInput) -> dict[str, dict[str, object]]:
        """Build one combined route, tool, and strategy questionnaire."""
        agent_criteria = {
            agent: _AGENT_DESCRIPTIONS.get(agent, f"Rota {agent} disponível no Midas.")
            for agent in state.available_agents
        }
        questions: dict[str, dict[str, object]] = {
            "agent": {
                "type": "choice",
                "instructions": (
                    "Escolha a rota que melhor atende à mensagem atual e às "
                    "capacidades resumidas no contexto. Use default para respostas "
                    "gerais sem necessidade de especialista."
                ),
                "criteria": agent_criteria,
            },
            "needs_mcp": {
                "type": "noul",
                "instructions": (
                    "A tarefa exige consultar uma ou mais ferramentas MCP listadas "
                    "para responder corretamente?"
                ),
            },
            "needs_analytics": {
                "type": "noul",
                "instructions": (
                    "A tarefa exige uma consulta analítica estruturada além das "
                    "ferramentas listadas? Respeite a disponibilidade informada."
                ),
            },
        }
        for index, tool_name in enumerate(state.available_tools):
            questions[f"tool_{index}"] = {
                "type": "noul",
                "instructions": (
                    f"Usar a ferramenta {tool_name} melhoraria materialmente a "
                    "resposta desta tarefa? "
                    f"Descrição: {_TOOL_DESCRIPTIONS.get(tool_name, 'Ferramenta autorizada.') }"
                ),
            }
        return questions

    @staticmethod
    def _probability(answer: object, name: str) -> float:
        """Read one bounded probability from a System One answer."""
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise InvalidDecisionError(f"Jev omitted the {name} decision")
        probability = answer.get("noul")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise InvalidDecisionError(f"Jev returned invalid {name} probability")
        return float(probability)

    @classmethod
    def _parse_response(
        cls,
        body: object,
        state: DecisionInput,
    ) -> ProviderResult:
        """Convert the provider response to locally validated decision data."""
        if not isinstance(body, dict):
            raise InvalidDecisionError("Jev response must be an object")
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise InvalidDecisionError("Jev response has no answers object")
        route_answer = answers.get("agent")
        if (
            not isinstance(route_answer, dict)
            or route_answer.get("type") != "choice"
            or route_answer.get("choice") not in state.available_agents
        ):
            raise InvalidDecisionError("Jev returned an unavailable agent")
        confidence = route_answer.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise InvalidDecisionError("Jev returned invalid route confidence")

        selected_tools = []
        for index, tool_name in enumerate(state.available_tools):
            if cls._probability(answers.get(f"tool_{index}"), tool_name) >= 0.5:
                selected_tools.append(tool_name)
        needs_mcp = cls._probability(answers.get("needs_mcp"), "MCP") >= 0.5
        needs_analytics = cls._probability(
            answers.get("needs_analytics"), "analytics"
        ) >= 0.5
        usage = body.get("usage")
        if not isinstance(usage, dict):
            raise InvalidDecisionError("Jev response has no usage object")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            not isinstance(input_tokens, int)
            or input_tokens < 0
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise InvalidDecisionError("Jev returned invalid token usage")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise InvalidDecisionError("Jev response has no model name")
        try:
            decision = MidasDecision(
                agent=route_answer["choice"],
                tools=selected_tools,
                needs_mcp=needs_mcp,
                needs_analytics=needs_analytics,
                confidence=float(confidence),
            )
            return ProviderResult(
                decision=decision,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except ValidationError as error:
            raise InvalidDecisionError("Jev decision failed local validation") from error
