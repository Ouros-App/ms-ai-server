import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from pymongo import AsyncMongoClient

from app.agents.graph import build_graph
from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.repositories.checkpointer import get_checkpointer
from app.repositories.memory import UserMemoryStore

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria o checkpointer MongoDB e o grafo durante a vida da API."""
    logger.info("application_starting")
    memory_client = AsyncMongoClient(settings.mongodb_uri)
    memory_store = UserMemoryStore(memory_client[settings.mongodb_database])
    await memory_store.ensure_indexes()
    logger.info("memory_indexes_ready database=%s", settings.mongodb_database)
    try:
        with get_checkpointer() as checkpointer:
            app.state.graph = build_graph(checkpointer, memory_store=memory_store)
            logger.info("application_ready")
            yield
    finally:
        await memory_client.close()
        logger.info("application_stopped")


app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def log_requests(request, call_next):
    """Registra ciclo, status e duracao sem registrar corpo ou headers sensiveis."""
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            (perf_counter() - started_at) * 1000,
        )
        raise

    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started_at) * 1000,
    )
    return response
