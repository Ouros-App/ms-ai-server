# AI Server

<!-- REPO-METADATA:START -->
<div align="center">

[![Repo Size](https://img.shields.io/github/repo-size/Ouros-App/ms-ai-server?style=flat-square&label=REPO%20SIZE)](https://github.com/Ouros-App/ms-ai-server)
[![Languages](https://img.shields.io/github/languages/count/Ouros-App/ms-ai-server?style=flat-square&label=LANGUAGES)](https://github.com/Ouros-App/ms-ai-server/languages)
[![Forks](https://img.shields.io/github/forks/Ouros-App/ms-ai-server?style=flat-square&label=FORKS)](https://github.com/Ouros-App/ms-ai-server/network/members)
[![Issues](https://img.shields.io/github/issues/Ouros-App/ms-ai-server?style=flat-square&label=ISSUES)](https://github.com/Ouros-App/ms-ai-server/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Ouros-App/ms-ai-server?style=flat-square&label=PULL%20REQUESTS)](https://github.com/Ouros-App/ms-ai-server/pulls)

</div>
<!-- REPO-METADATA:END -->

API FastAPI de orquestração de agentes de IA com LangGraph. O serviço usa MongoDB para persistir o estado das conversas e as memórias dos usuários, e pode chamar provedores Groq e NVIDIA NIM.

## Status e escopo

O serviço está implementado com:

- endpoints de saúde, chat e histórico de conversas;
- endpoint Prometheus autenticado em `/metrics`;
- roteamento entre os agentes `faq`, `sustainability`, `ranking`, `support` e `fallback`;
- especialistas com contrato JSON e um agente `default` dedicado à síntese da resposta;
- resposta padrão quando nenhum provedor de IA está configurado;
- autenticação de usuário exclusivamente por JWT RS256 do Keycloak validado localmente via JWKS;
- guardrails de entrada e revisão de saída.

As chaves de IA são opcionais para iniciar a aplicação. Todos os endpoints autenticados aceitam somente JWTs emitidos pelo realm `ouros`.

## Principais componentes

- `app/main.py`: cria a aplicação FastAPI, inicializa MongoDB, o checkpointer e o grafo.
- `app/api/routes.py`: expõe as rotas HTTP.
- `app/agents/graph.py`: define o fluxo de roteamento, execução estruturada e síntese.
- `app/agents/mcp.py`: encaminha o JWT validado do usuário ao MCP e aplica a allowlist.
- `app/agents/prompts.py`: regras comuns, contrato JSON, rota e prompts especializados.
- `app/agents/model.py`: configura os perfis rápido e potente do Groq e NVIDIA NIM.
- `app/agents/guardrails.py`: valida entradas e revisa respostas.
- `app/repositories/`: checkpointer, memórias e posse das threads.
- `app/services/`: execução do chat e leitura do histórico.
- `tests/`: testes da API, configuração, grafo, guardrails, modelos, prompts, threads e tools.

## Pré-requisitos

- Docker e Docker Compose para a execução completa com MongoDB.
- Python 3.12 para execução fora do container.
- Configure `AUTH_JWT_ISSUER`, `AUTH_JWT_AUDIENCE` e `AUTH_JWKS_URL` do Keycloak e use um JWT de usuário nos endpoints autenticados.
- Chaves `GROQ_API_KEY` e/ou `NVIDIA_API_KEY` quando a resposta por IA for necessária.

## Instalação e configuração

Copie `.env.example` para `.env` e preencha os valores necessários. O arquivo de exemplo documenta:

| Variável | Função |
| --- | --- |
| `APP_PORT` | Porta publicada pelo Compose, com padrão `8000`. |
| `METRICS_TOKEN` | Bearer dedicado ao scrape de `/metrics`; não reutilize JWT de usuário. |
| `INFISICAL_TOKEN` / `INFISICAL_PROJECT_ID` / `INFISICAL_ENV` / `INFISICAL_PATH` | Bootstrap opcional do Infisical. Configure as quatro juntas; `INFISICAL_ENV` aceita `prod` ou `dev`. |
| `INFISICAL_HOST` | Host do Infisical, padrão `https://app.infisical.com`. |
| `MONGODB_URI` | URI do MongoDB para execução fora do Compose. |
| `MONGODB_URI_DOCKER` | URI usada pelo serviço no Compose. |
| `MONGODB_DATABASE` | Banco usado pelo serviço, com padrão `mongodb-ai-prod`; use `mongodb-ai-qa` no ambiente de QA. |
| `GROQ_API_KEY` / `GROQ_FAST_MODEL` / `GROQ_MODEL` | Provedor Groq e perfis rápido/potente. |
| `NVIDIA_API_KEY` / `NVIDIA_NIM_FAST_MODEL` / `NVIDIA_NIM_MODEL` / `NVIDIA_NIM_BASE_URL` | Provedor NVIDIA NIM e perfis rápido/potente. |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT_SECONDS` | Parâmetros das chamadas ao modelo. |
| `MCP_TOOL_TIMEOUT_SECONDS` | Limite por chamada de tool, evitando que uma dependência consuma todo o tempo da requisição. |
| `AUTH_JWT_ISSUER` / `AUTH_JWT_AUDIENCE` / `AUTH_JWKS_URL` | Contrato oficial do Keycloak; valida assinatura RS256, issuer, audience e expiração. `database_id` é o ID do banco legado e `sub` permanece a identidade do Keycloak. |
| `MCP_URL` | Endpoint Streamable HTTP do servidor MCP externo. |
| `MCP_RESOURCE_URL` | Identificador/URL do recurso MCP. |
| `MCP_TOOLS_CACHE_TTL_SECONDS` | TTL do cache de tools MCP por token validado. |

Não versione o arquivo `.env` nem os tokens. Quando o Infisical está totalmente configurado, os secrets carregados do cofre são aplicados antes da criação de `Settings` e prevalecem sobre valores locais com a mesma chave. Sem nenhuma das quatro variáveis de bootstrap, o serviço pode rodar em modo local. Configuração parcial ou ambiente inválido interrompe o startup para evitar fallback silencioso.

Em deploy, `INFISICAL_TOKEN` é o único bootstrap secreto que precisa existir fora do cofre. Os secrets de aplicação esperados no Infisical incluem `MONGODB_URI` quando contiver credenciais, `GROQ_API_KEY` e `NVIDIA_API_KEY`. A autenticação de usuário não depende mais de shared secrets locais.

O roteador, guardrails, FAQ, suporte, fallback e o sintetizador default usam os perfis rápidos
`GROQ_FAST_MODEL` e `NVIDIA_NIM_FAST_MODEL`. Ranking e sustentabilidade usam os
perfis potentes `GROQ_MODEL` e `NVIDIA_NIM_MODEL`. Se o
Groq falhar, o NVIDIA NIM é usado como fallback do mesmo perfil.

O fluxo é `router → especialistas → fan-in → default`. O roteador pode selecionar
até quatro especialistas independentes, que rodam em paralelo no LangGraph. Cada
especialista retorna somente JSON com fatos, recomendações, dados ausentes e
fontes; o `default` é o único agente que gera linguagem natural para o usuário.

Quando um especialista retorna `needs_input`, o fan-in persiste a rota e os dados
faltantes como estado estruturado da conversa, separados por especialista e com a
exigência de dados autenticados preservada. Respostas curtas como "30 dias", "1 ciclo"
ou "na minha fazenda" continuam a tarefa anterior sem depender apenas de palavras-chave
ou de o roteador reconstruir a intenção do zero. Saudações e perguntas como
"quem é você?" usam fast paths determinísticos e não gastam chamadas de LLM; um
cancelamento limpa a tarefa, e uma troca explícita para outro domínio aposenta
pendências antigas para elas não reaparecerem mais tarde.

As tools de memória e MCP ficam disponíveis somente para especialistas. O cliente
MCP usa Streamable HTTP, troca o JWT validado por um token delegado de backend,
aplica uma allowlist mínima por especialista e faz filtragem adicional dos resultados
escopados em `app/agents/mcp.py`. O sintetizador não recebe nenhuma dessas tools.
Sustentabilidade usa preferencialmente `get_consumption_summary(period_days)`: quando
uma consulta pessoal contém um período explícito, o backend extrai essa janela e faz
o prefetch determinístico do resumo antes do modelo, evitando carregar registros
brutos desnecessários. `farm_id`, `user_id` e SQL arbitrário não viram argumentos
controlados pelo modelo. Ranking consulta a base de conhecimento para regras mutáveis
e não recebe dados brutos que não consigam provar posição/classificação. Resultados
de tools são serializados como JSON com limite de tamanho, cada chamada possui timeout
próprio e falhas são degradadas para um erro seguro, sem vazar exceções ou derrubar
toda a conversa. O Request Trace registra apenas metadados estruturais dos resultados
de tools, nunca o conteúdo operacional retornado pela fazenda.

## Execução

Com Docker Compose:

```bash
docker compose up --build
```

A documentação interativa fica em [http://localhost:8000/docs](http://localhost:8000/docs).

Para execução direta, instale as dependências de `requirements.txt` e use o entrypoint `app.main:app` com Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Uso da API

Rotas públicas:

- `GET /`: confirma que o serviço está em execução.
- `GET /health`: retorna `{"status":"ok"}`.
- `/docs`, `/redoc` e `/openapi.json`: documentação da API.

Rotas autenticadas por JWT de usuário:

- `POST /v1/chat`: processa uma mensagem.
- `GET /v1/chat/{thread_id}/history`: retorna o histórico paginado de uma conversa.

Exemplo de requisição:

```json
{
  "message": "Como funciona o ranking?",
  "thread_id": "conversa-1"
}
```

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <jwt-do-usuario>" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"conversa-1","message":"Como funciona o ranking?"}'
```

A resposta contém `thread_id`, `message`, `agents` e `tools`. A identidade efetiva vem sempre do claim assinado `database_id`; `user_id`, quando enviado por clientes antigos, é apenas um campo de compatibilidade e precisa coincidir com o JWT. O `sub` continua sendo a identidade estável do Keycloak. O primeiro chat vincula o `thread_id` ao usuário autenticado e essa posse é imutável.

O histórico aceita `limit` entre 1 e 100, com padrão 20, e o cursor `before` para buscar a página anterior:

```bash
curl "http://localhost:8000/v1/chat/conversa-1/history?limit=20" \
  -H "Authorization: Bearer <token-configurado>"
```

## Observabilidade

O endpoint `GET /metrics` expõe somente métricas agregadas de baixa cardinalidade.
Ele usa um `METRICS_TOKEN` dedicado ao coletor e não depende de um JWT de usuário,
o que permite scrapes recorrentes sem criar uma sessão privilegiada artificial.

Além de HTTP, roteamento, agentes e tools, o serviço mede:

- tempo ponta a ponta de cada mensagem do Midas por resultado;
- duração e quantidade de chamadas LLM por perfil/modelo;
- tokens de entrada, cache de entrada e saída reportados pelo provedor;
- tempo e resultado das chamadas do AI Server ao Knowledge MCP;
- tarefas conversacionais pendentes e origem da decisão de rota.

Os contadores de tokens representam uso técnico. O AI Server **não calcula dólares**:
a precificação fica no serviço de Telemetry para que mudanças de tabela de preço não
alterem a instrumentação e para distinguir modelos cobrados por token de modelos NIM
precificados por infraestrutura.

Nenhuma métrica usa `user_id`, `farm_id`, `thread_id`, request ID ou conteúdo
do prompt como label.

```yaml
scrape_configs:
  - job_name: ms-ai-server
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials: <metrics-token>
    static_configs:
      - targets: ["ms-ai-server.discloud.app"]
```

## Testes e qualidade

Os mesmos comandos usados no CI são:

```bash
ruff check .
pytest --cov=app --cov-report=xml:coverage.xml
python -m compileall .
```

O workflow também executa SonarCloud e CodeQL.

## Licença

Este projeto está sob a licença MIT, conforme o arquivo [LICENSE](LICENSE).


## Principais contribuidores

<!-- CONTRIBUTORS:START -->
- [@Nicolas25vlad](https://github.com/Nicolas25vlad) — 4 contribuições
<!-- CONTRIBUTORS:END -->

> Atualizado automaticamente semanalmente pelo workflow de metadados do README.


## Debug Console

O AI Server inclui uma UI de diagnóstico opcional em `/debug`. Ela não faz parte
do produto final e fica desligada por padrão.

Ative explicitamente no ambiente de debug:

```dotenv
DEBUG_UI_ENABLED=true
DEBUG_UI_KEYCLOAK_TOKEN_URL=https://ouros-keycloak.discloud.app/realms/ouros/protocol/openid-connect/token
DEBUG_UI_KEYCLOAK_CLIENT_ID=ms-ai-server-debug
DEBUG_UI_KEYCLOAK_CLIENT_SECRET=<secret-gerado-pelo-keycloak>
DEBUG_UI_COOKIE_SECURE=true
DEBUG_UI_REQUEST_TIMEOUT_SECONDS=8
```

Em desenvolvimento local via HTTP, use `DEBUG_UI_COOKIE_SECURE=false`.

O login usa uma conta real já existente no Ouros (`farm_owner`,
`company_employee` ou `admin`). O backend do AI Server troca as credenciais
diretamente no token endpoint do Keycloak usando o client confidencial e isolado
`ms-ai-server-debug`. Esse client é a única exceção de Direct Access Grant para
o painel de desenvolvimento; mobile e web continuam obrigados ao Browser Flow
com Authorization Code + PKCE e, quando habilitado, OTP por e-mail.

O browser nunca recebe o client secret nem refresh tokens. Access e refresh token
ficam somente em cookies `HttpOnly`, `SameSite=Strict`, restritos a `/debug`.
Com isso o Debug Console deixa de depender do broker legado do `ms-auth-service`,
permitindo desativar `KEYCLOAK_PASSWORD_BROKER_ENABLED` sem quebrar o painel.

A interface usa a mesma pipeline de chat do endpoint normal. Para cada resposta
ela mostra:

- Markdown GFM renderizado e sanitizado;
- loading enquanto o grafo está executando;
- rota escolhida pelo router;
- agentes executados;
- tools chamadas;
- resultados estruturados dos especialistas;
- respostas internas dos agentes capturadas durante aquela requisição;
- trace cronológico com guardrail, router, agentes, tools e duração.

O coletor de trace é request-local e vira no-op fora das requisições do painel.
Conversas e títulos da sidebar ficam somente no `localStorage` do navegador; o
token de autenticação não é armazenado em JavaScript.

O frontend é HTML/CSS/JavaScript vanilla em `app/debug_ui/static` e o backend
específico da UI fica em `app/debug_ui`, mantendo o painel desacoplado das
rotas públicas do produto.
