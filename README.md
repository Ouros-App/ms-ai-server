# FastAPI Microservice Template

Template minimo para iniciar um microservico com FastAPI.

## Estrutura

```text
.
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models/
│   │   └── __init__.py
│   ├── repositories/
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── common.py
│   ├── services/
│   │   └── __init__.py
│   └── main.py
├── tests/
│   └── __init__.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Pastas principais

- `app/main.py`: cria a aplicacao FastAPI e registra as rotas.
- `app/api/`: rotas e agrupamento de endpoints.
- `app/core/`: configuracoes centrais do servico.
- `app/schemas/`: contratos de entrada e saida com Pydantic.
- `app/services/`: regras de negocio.
- `app/repositories/`: acesso a dados ou integracoes externas.
- `app/models/`: modelos internos ou modelos de banco, quando existirem.
- `tests/`: testes automatizados.

## Rotas

- `GET /` retorna uma mensagem simples da aplicacao.
- `GET /health` retorna o status de saude do servico.

## Rodando localmente

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Inicie a API:

```bash
uvicorn app.main:app --reload
```

Acesse:

- API: `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs`

## Rodando com Docker Compose

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Suba o servico:

```bash
docker compose up --build
```

Para parar:

```bash
docker compose down
```

## Variaveis de ambiente

| Nome | Padrao | Descricao |
| --- | --- | --- |
| `APP_PORT` | `8000` | Porta publicada no host pelo Docker Compose. |
