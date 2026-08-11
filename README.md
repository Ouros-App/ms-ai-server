# AI Server

API principal para os fronts consumirem agentes de IA. A base ja possui FastAPI, LangGraph e memoria persistente no MongoDB; os agentes, prompts e tools ainda sao propositalmente genericos.

## Estrutura

- `app/agents/graph.py`: fluxo `router -> agente` e registro de agentes.
- `app/agents/prompts.py`: prompts para preencher.
- `app/agents/tools.py`: tools permitidas para preencher.
- `app/services/chat.py`: entrada unica do grafo.
- MongoDB: checkpoints por `thread_id`, incluindo mensagens e estado do grafo.

## Rodar

```bash
docker compose up --build
```

Copie `.env.example` para `.env` apenas se precisar alterar os valores padrao.

API: `http://localhost:8000/docs`

```json
POST /v1/chat
{
  "user_id": "usuario-1",
  "message": "teste"
}
```

Enquanto nenhum agente for definido, a rota retorna uma resposta de placeholder. Para adicionar um agente, crie o no em `app/agents/graph.py`, inclua-o em `AGENTS` e altere `route_request`. O mesmo `thread_id` so pode ser reutilizado pelo `user_id` que o criou.
