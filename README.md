# AI Server

API FastAPI de orquestração de agentes de IA com LangGraph. O serviço usa MongoDB para persistir o estado das conversas e as memórias dos usuários, e pode chamar provedores Groq e NVIDIA NIM.

## Status e escopo

O serviço está implementado com:

- endpoints de saúde, chat e histórico de conversas;
- roteamento entre os agentes `faq`, `sustainability`, `ranking`, `support` e `fallback`;
- resposta padrão quando nenhum provedor de IA está configurado;
- autenticação por um token Bearer compartilhado;
- guardrails de entrada e revisão de saída.

As chaves de IA são opcionais para iniciar a aplicação, mas o `AUTH_BEARER_TOKEN` é necessário para acessar os endpoints autenticados.

## Principais componentes

- `app/main.py`: cria a aplicação FastAPI, inicializa MongoDB, o checkpointer e o grafo.
- `app/api/routes.py`: expõe as rotas HTTP.
- `app/agents/graph.py`: define o fluxo de roteamento e execução dos agentes.
- `app/agents/prompts.py`: regras comuns, rotas e prompts especializados.
- `app/agents/model.py`: configura Groq e NVIDIA NIM, com fallback quando ambos estão disponíveis.
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
| `GROQ_API_KEY` / `GROQ_MODEL` | Provedor e modelo Groq. |
| `NVIDIA_API_KEY` / `NVIDIA_NIM_MODEL` / `NVIDIA_NIM_BASE_URL` | Provedor NVIDIA NIM. |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT_SECONDS` | Parâmetros das chamadas ao modelo. |
| `AUTH_BEARER_TOKEN` | Token exigido no header `Authorization: Bearer ...`. |

Não versione o arquivo `.env` nem os tokens.

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

Rotas autenticadas:

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
