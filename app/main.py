from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.mongodb import MongoDBSaver

from app.agents.graph import build_graph
from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria o checkpointer MongoDB e o grafo durante a vida da API."""
    with MongoDBSaver.from_conn_string(
        settings.mongodb_uri,
        db_name=settings.mongodb_database,
    ) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
)
app.include_router(router)
