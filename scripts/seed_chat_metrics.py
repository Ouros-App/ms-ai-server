#!/usr/bin/env python3
"""Generate authenticated chat traffic for Prometheus smoke testing."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_AUTH_URL = "https://ms-auth-service.discloud.app/v1/auth/token"
DEFAULT_CHAT_URL = "https://ms-ai-server.discloud.app/v1/chat"
TOKEN_BATCH_SIZE = 10
MAX_RATE_LIMIT_RETRIES = 3
DEFAULT_RETRY_AFTER_SECONDS = 300
PROMPTS = (
    "Como acompanhar o consumo de água em cada fase da produção?",
    "Quais indicadores ajudam a perceber desperdício de água?",
    "Como comparar leituras semanais de forma justa?",
    "Que causas podem explicar aumento no consumo de energia?",
    "Como acompanhar eficiência energética ao longo de um ciclo?",
    "Quais dados históricos são essenciais para analisar tendências?",
    "Como identificar lacunas ou inconsistências em medições?",
    "O que considerar ao comparar duas épocas do ano?",
    "Como avaliar se uma ação de economia teve efeito?",
    "Quais cuidados tomar ao interpretar um indicador isolado?",
    "Como resumir tendências de consumo em linguagem simples?",
    "Que gráfico ajuda a visualizar mudanças de consumo por período?",
    "Como escolher um período de referência para comparação?",
    "Como consumo de água se relaciona com sustentabilidade?",
    "Quais indicadores podem compor uma análise mensal da granja?",
    "Como distinguir uma oscilação normal de um sinal de alerta?",
    "Que registros ajudam a contextualizar uma mudança no histórico?",
    "Como acompanhar consumo de energia sem comparar ciclos diferentes?",
    "Quais perguntas revisar antes de concluir que houve melhora?",
    "Como apresentar indicadores para facilitar acompanhamento?",
    "Que fatores operacionais podem influenciar o uso de água?",
    "Como verificar se há tendência de queda no consumo?",
    "Quais práticas apoiam o uso eficiente de recursos?",
    "Como evitar conclusões precipitadas ao comparar granjas?",
    "Faça um checklist curto para revisar água, energia e sustentabilidade.",
)


def post_json(
    url: str, payload: dict[str, Any], token: str | None = None
) -> tuple[int, dict[str, Any], Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            status, raw, response_headers = response.status, response.read(), response.headers
    except HTTPError as error:
        status, raw, response_headers = error.code, error.read(), error.headers
    except URLError as error:
        raise RuntimeError(f"Request failed: {error.reason}") from error
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {}
    return status, body if isinstance(body, dict) else {}, response_headers


def post_with_backoff(
    url: str, payload: dict[str, Any], token: str | None = None
) -> tuple[int, dict[str, Any]]:
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        status, body, headers = post_json(url, payload, token)
        if status != 429 or attempt == MAX_RATE_LIMIT_RETRIES - 1:
            return status, body
        try:
            delay = max(1, int(headers.get("Retry-After", DEFAULT_RETRY_AFTER_SECONDS)))
        except (TypeError, ValueError):
            delay = DEFAULT_RETRY_AFTER_SECONDS
        print(f"Rate limited; retrying in {delay}s", file=sys.stderr, flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def authenticate(account: dict[str, str], auth_url: str) -> str:
    status, body = post_with_backoff(
        auth_url,
        {"email": account["email"], "password": account["password"]},
    )
    token = body.get("access_token")
    if status != 200 or not isinstance(token, str):
        label = account.get("label", "account")
        raise RuntimeError(f"Login failed for {label}: HTTP {status}")
    return token


def load_accounts(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Accounts file must contain a non-empty JSON array")
    for index, account in enumerate(data, start=1):
        if not isinstance(account, dict) or not all(
            isinstance(account.get(key), str) and account[key]
            for key in ("email", "password")
        ):
            raise ValueError(f"Account {index} must include non-empty email and password")
    return data


def seed_account(
    account: dict[str, str], calls: int, auth_url: str, chat_url: str
) -> None:
    token = ""
    for call_number in range(calls):
        if call_number % TOKEN_BATCH_SIZE == 0:
            token = authenticate(account, auth_url)
        payload = {
            "thread_id": str(uuid.uuid4()),
            "message": PROMPTS[call_number % len(PROMPTS)],
        }
        status, body = post_with_backoff(chat_url, payload, token)
        if status == 401:
            token = authenticate(account, auth_url)
            status, body = post_with_backoff(chat_url, payload, token)
        if status != 200 or not isinstance(body.get("message"), str):
            label = account.get("label", "account")
            raise RuntimeError(f"Chat failed for {label}, call {call_number + 1}: HTTP {status}")
    print(f"{account.get('label', 'account')}: {calls} chats accepted", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounts-file", type=Path, required=True)
    parser.add_argument("--calls-per-account", type=int, default=10)
    parser.add_argument("--auth-url", default=DEFAULT_AUTH_URL)
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL)
    args = parser.parse_args()
    if args.calls_per_account < 10:
        parser.error("--calls-per-account must be at least 10")
    try:
        accounts = load_accounts(args.accounts_file)
        for account in accounts:
            seed_account(account, args.calls_per_account, args.auth_url, args.chat_url)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
