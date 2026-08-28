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
- autenticação por um token Bearer compartilhado;
- guardrails de entrada e revisão de saída.

As chaves de IA são opcionais para iniciar a aplicação, mas o `AUTH_BEARER_TOKEN` é necessário para acessar os endpoints autenticados.

## Principais componentes

- `app/main.py`: cria a aplicação FastAPI, inicializa MongoDB, o checkpointer e o grafo.
- `app/api/routes.py`: expõe as rotas HTTP.
- `app/agents/graph.py`: define o fluxo de roteamento, execução estruturada e síntese.
- `app/agents/mcp.py`: conecta o MCP externo, emite JWTs curtos e aplica a allowlist.
- `app/agents/prompts.py`: regras comuns, contrato JSON, rota e prompts especializados.
- `app/agents/model.py`: configura os perfis rápido e potente do Groq e NVIDIA NIM.
- `app/agents/guardrails.py`: valida entradas e revisa respostas.
- `app/repositories/`: checkpointer, memórias e posse das threads.
- `app/services/`: execução do chat e leitura do histórico.
- `tests/`: testes da API, configuração, grafo, guardrails, modelos, prompts, threads e tools.

## Pré-requisitos

- Docker e Docker Compose para a execução completa com MongoDB.
- Python 3.12 para execução fora do container.
- Um token para `AUTH_BEARER_TOKEN`.
- Chaves `GROQ_API_KEY` e/ou `NVIDIA_API_KEY` quando a resposta por IA for necessária.

## Instalação e configuração

Copie `.env.example` para `.env` e preencha os valores necessários. O arquivo de exemplo documenta:

| Variável | Função |
| --- | --- |
| `APP_PORT` | Porta publicada pelo Compose, com padrão `8000`. |
| `MONGODB_URI` | URI do MongoDB para execução fora do Compose. |
| `MONGODB_URI_DOCKER` | URI usada pelo serviço no Compose. |
| `MONGODB_DATABASE` | Banco usado pelo serviço, com padrão `ai_server`. |
| `GROQ_API_KEY` / `GROQ_FAST_MODEL` / `GROQ_MODEL` | Provedor Groq e perfis rápido/potente. |
| `NVIDIA_API_KEY` / `NVIDIA_NIM_FAST_MODEL` / `NVIDIA_NIM_MODEL` / `NVIDIA_NIM_BASE_URL` | Provedor NVIDIA NIM e perfis rápido/potente. |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT_SECONDS` | Parâmetros das chamadas ao modelo. |
| `AUTH_BEARER_TOKEN` | Token exigido no header `Authorization: Bearer ...`. |
| `MCP_URL` | Endpoint Streamable HTTP do servidor MCP externo. |
| `MCP_ACCESS_TOKEN` | Token MCP fixo de fallback; prefira JWT por usuário em produção. |
| `MCP_JWT_SECRET` / `MCP_JWT_ISSUER_URL` / `MCP_RESOURCE_URL` | Emissão de JWT curto por usuário para o MCP. |
| `MCP_USER_TYPE` / `MCP_JWT_TTL_SECONDS` | Identidade e validade do JWT MCP. |

Não versione o arquivo `.env` nem os tokens.

O roteador, guardrails, FAQ, suporte, fallback e o sintetizador default usam os perfis rápidos
`GROQ_FAST_MODEL` e `NVIDIA_NIM_FAST_MODEL`. Ranking e sustentabilidade usam os
perfis potentes `GROQ_MODEL` e `NVIDIA_NIM_MODEL`. Se o
Groq falhar, o NVIDIA NIM é usado como fallback do mesmo perfil.

O fluxo é `router → especialistas → fan-in → default`. O roteador pode selecionar
até quatro especialistas independentes, que rodam em paralelo no LangGraph. Cada
especialista retorna somente JSON com fatos, recomendações, dados ausentes e
fontes; o `default` é o único agente que gera linguagem natural para o usuário.

As tools de memória e MCP ficam disponíveis somente para especialistas. O cliente
MCP usa Streamable HTTP, cria um JWT curto por usuário quando `MCP_JWT_SECRET` está
configurado e aplica a allowlist em `app/agents/mcp.py`. O sintetizador não recebe
nenhuma dessas tools.

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

- `/docs`, `/redoc` e `/openapi.json`: documentação da API.

Rotas autenticadas:

- `GET /`: confirma que o serviço está em execução.
- `GET /health`: retorna `{"status":"ok"}`.
- `POST /v1/chat`: processa uma mensagem.
- `GET /v1/chat/{thread_id}/history`: retorna o histórico paginado de uma conversa.

Exemplo de requisição:

```json
{
  "user_id": "usuario-1",
  "message": "Como funciona o ranking?",
  "thread_id": "conversa-1"
}
```

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <token-configurado>" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usuario-1","thread_id":"conversa-1","message":"Como funciona o ranking?"}'
```

A resposta contém `thread_id`, `message`, `agents` e `tools`. O mesmo `thread_id` não pode ser usado por outro `user_id`.

O histórico aceita `limit` entre 1 e 100, com padrão 20, e o cursor `before` para buscar a página anterior:

```bash
curl "http://localhost:8000/v1/chat/conversa-1/history?user_id=usuario-1&limit=20" \
  -H "Authorization: Bearer <token-configurado>"
```

## Observabilidade

O endpoint `GET /metrics` expõe métricas Prometheus de requisições HTTP, latência,
status, resultados do chat, agentes e tools utilizadas. Ele exige o mesmo Bearer
Token da API. Configure o Prometheus para enviar esse token no scrape e use o
Prometheus como datasource no Grafana:

```yaml
scrape_configs:
  - job_name: ms-ai-server
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials: <token-configurado>
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
