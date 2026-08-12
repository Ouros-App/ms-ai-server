# AI Server

API principal para os fronts consumirem agentes de IA. A base ja possui FastAPI, LangGraph e memoria persistente no MongoDB; os agentes, prompts e tools ainda sao propositalmente genericos.

## Estrutura

- `app/agents/graph.py`: fluxo `router -> agente` e registro de agentes.
- `app/agents/model.py`: Groq como provider principal e NVIDIA NIM como fallback.
- `app/agents/prompts.py`: prompts para preencher.
- `app/agents/tools.py`: tools permitidas para preencher.
- `app/services/chat.py`: entrada unica do grafo.
- MongoDB: checkpoints por `thread_id`, incluindo mensagens e estado do grafo.

## Rodar

Sem chaves de IA, o Compose sobe a resposta de placeholder:

```bash
docker compose up --build
```

Para ativar a IA, copie `.env.example` para `.env` e preencha `GROQ_API_KEY` e `NVIDIA_API_KEY` antes de executar o Compose. O Groq recebe cada chamada primeiro; se falhar, LangChain reexecuta a mesma chamada no endpoint NIM configurado por `NVIDIA_NIM_BASE_URL`. `MONGODB_URI` serve para execucao no host; o Compose usa `MONGODB_URI_DOCKER`.

API: `http://localhost:8000/docs`

`POST /v1/chat`

```json
{
  "user_id": "usuario-1",
  "message": "teste"
}
```

Enquanto nenhum agente for definido, a rota retorna uma resposta de placeholder. Para adicionar um agente, crie o no em `app/agents/graph.py`, inclua-o em `AGENTS` e altere `route_request`. O mesmo `thread_id` so pode ser reutilizado pelo `user_id` que o criou.
