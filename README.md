# AI Server

API principal para os fronts consumirem agentes de IA. A base possui FastAPI,
LangGraph, roteamento semantico, agentes especializados e memoria persistente
no MongoDB. RAG e MCP ficam fora deste servico e serao integrados por outro
microservico quando necessario.

## Estrutura

- `app/agents/graph.py`: fluxo `router -> agente` e registro dos agentes.
- `app/agents/model.py`: Groq como provider principal e NVIDIA NIM como fallback.
- `app/agents/prompts.py`: regras comuns, roteamento e prompts dos agentes.
- `app/agents/guardrails.py`: bloqueios de prompt injection e vazamento de credenciais.
- `app/agents/tools.py`: tools permitidas para preencher.
- `app/services/chat.py`: entrada unica do grafo.
- MongoDB: checkpoints por `thread_id` e colecao `user_memories` para memorias persistentes por usuario.

## Rodar

Sem chaves de IA, o Compose sobe a resposta de placeholder:

```bash
docker compose up --build
```

Para ativar a IA, copie `.env.example` para `.env` e preencha `GROQ_API_KEY` e `NVIDIA_API_KEY` antes de executar o Compose. O Groq recebe cada chamada primeiro; se falhar, LangChain reexecuta a mesma chamada no endpoint NIM configurado por `NVIDIA_NIM_BASE_URL`. `MONGODB_URI` serve para execucao no host; o Compose usa `MONGODB_URI_DOCKER`.

`LLM_TOTAL_TIMEOUT_SECONDS` limita o tempo total de roteamento e resposta. Quando
o provider demora além desse limite, a API retorna `503` controlado para o
consumidor tentar novamente, evitando um `504` do gateway.

O sistema usa perfis de LLM: `GROQ_FAST_MODEL` para roteamento, guardrails, FAQ,
suporte e fallback; `GROQ_MODEL` para ranking, sustentabilidade e respostas que
exigem mais raciocinio. Se o Groq falhar, o NVIDIA NIM configurado continua como
fallback.

API: `http://localhost:8000/docs`

Todos os endpoints da API exigem o token Bearer unico configurado em
`AUTH_BEARER_TOKEN`, incluindo `/`, `/health`, `/v1/chat` e o historico. Os
endpoints de documentacao (`/docs`, `/redoc` e `/openapi.json`) permanecem
publicos para o Swagger funcionar normalmente.

Gere um token local com `python -c "import secrets; print(secrets.token_urlsafe(32))"`
e preencha `AUTH_BEARER_TOKEN` no `.env`. O token fica apenas no backend e no
cliente autorizado; nao o commite no repositorio.

```json
{
  "user_id": "usuario-1",
  "message": "Qual foi minha ultima pergunta?",
  "thread_id": "conversa-1"
}
```

Exemplo de chamada autenticada:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <token-configurado>" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usuario-1","thread_id":"conversa-1","message":"Qual foi minha ultima pergunta?"}'
```

A resposta informa o `thread_id`, a mensagem gerada, os `agents` consultados e as
`tools` executadas naquela mensagem. Se nenhuma tool for usada, `tools` retorna
uma lista vazia:

```json
{
  "thread_id": "conversa-1",
  "message": "...",
  "agents": ["router", "default"],
  "tools": ["recall_user_memories"]
}
```

O `user_id` e usado como dono da memoria persistida. O `thread_id` identifica uma conversa especifica, entao o mesmo usuario pode ter varias conversas. O mesmo `thread_id` so pode ser reutilizado pelo mesmo `user_id`; outro usuario recebe `403`. Como o token Bearer e compartilhado neste projeto escolar, o backend confia no `user_id` enviado pelo cliente.

O agente pode usar as tools `recall_user_memories` e `save_user_memory` para
recuperar ou salvar memorias curtas entre conversas. O `user_id` e injetado pelo
backend nas tools e nao e escolhido pelo modelo.

Com um provider de IA configurado, o roteador escolhe uma rota entre `faq`,
`sustainability`, `ranking`, `support` e `fallback`, e o agente correspondente
responde com seu prompt especializado. Sem chaves de IA, a API continua
retornando o placeholder seguro de configuracao. O agente `default` permanece
como fallback tecnico e para registros customizados passados a `build_graph`.

Para carregar uma conversa existente, use o historico paginado:

```bash
curl "http://localhost:8000/v1/chat/conversa-1/history?user_id=usuario-1&limit=20" \
  -H "Authorization: Bearer <token-configurado>"
```

Quando houver mais mensagens, a resposta retorna `next_cursor`. Envie esse valor
como `before` na proxima chamada para buscar a pagina anterior. O limite aceito e
de 1 a 100 mensagens por requisicao.

Antes de chamar o modelo, o sistema bloqueia instrucoes internas, prompt injection,
credenciais, tokens, chaves, senhas, PII, pedidos perigosos ou ilicitos e assuntos
fora do escopo. Na saida, redige PII e segredos, limita o tamanho, rejeita respostas
vazias e substitui claims nao confirmados. Essas protecoes ocorrem antes e depois
do modelo; a deteccao de prompt injection, escopo e dados internos e principalmente
de entrada.

Os logs registram ciclo de inicializacao, requisições, bloqueios do guardrail,
tools utilizadas, status e duracao. O corpo das requisições, tokens e conteúdo
das memorias nao sao registrados.
