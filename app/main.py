import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from pymongo import AsyncMongoClient

from app.agents.graph import build_graph
from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.metrics import observe_http_request
from app.repositories.checkpointer import get_checkpointer
from app.repositories.memory import UserMemoryStore
from app.repositories.thread_ownership import ThreadOwnershipStore

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria o checkpointer MongoDB e o grafo durante a vida da API."""
    logger.info("application_starting")
    memory_client = None
    try:
        memory_client = AsyncMongoClient(settings.mongodb_uri)
        database = memory_client[settings.mongodb_database]
        memory_store = UserMemoryStore(database)
        thread_ownership = ThreadOwnershipStore(database)
        logger.info("database_ready database=%s", settings.mongodb_database)
        with get_checkpointer() as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.thread_ownership = thread_ownership
            app.state.graph = build_graph(checkpointer, memory_store=memory_store)
            logger.info("application_ready")
            yield
    finally:
        if memory_client is not None:
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
    safe_path = request.url.path.replace("\r", "\\r").replace("\n", "\\n")
    try:
        response = await call_next(request)
    except Exception:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        observe_http_request(
            request.method,
            route,
            500,
            perf_counter() - started_at,
        )
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.1f",
            request.method,
            safe_path,
            (perf_counter() - started_at) * 1000,
        )
        raise

    route = getattr(request.scope.get("route"), "path", "unmatched")
    observe_http_request(
        request.method,
        route,
        response.status_code,
        perf_counter() - started_at,
    )
    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        safe_path,
        response.status_code,
        (perf_counter() - started_at) * 1000,
    )
    return response
