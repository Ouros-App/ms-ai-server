from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.auth import (
    Principal,
    get_current_principal,
    require_metrics_bearer,
    user_id_for_request,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import HealthResponse, MessageResponse
from app.schemas.history import HistoryResponse
from app.services.chat import invoke_graph
from app.services.history import get_thread_history

router = APIRouter()


@router.get(
    "/",
    response_model=MessageResponse,
    dependencies=[Depends(get_current_principal)],
)
async def read_root() -> MessageResponse:
    """Informa que a API esta disponivel."""
    return MessageResponse(message="AI Server is running")


@router.get(
    "/health",
    response_model=HealthResponse,
    dependencies=[Depends(get_current_principal)],
)
async def health_check() -> HealthResponse:
    """Retorna o estado de saude da API."""
    return HealthResponse(status="ok")


@router.get("/metrics", include_in_schema=False)
async def metrics(
    _claims: Annotated[dict, Depends(require_metrics_bearer)],
) -> Response:
    """Expoe metricas no formato Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> ChatResponse:
    """Processa uma mensagem dentro de uma thread persistente."""
    user_id = user_id_for_request(payload.user_id, principal)
    return await invoke_graph(
        request.app.state.graph,
        payload.model_copy(update={"user_id": user_id}),
        user_id,
        getattr(request.app.state, "thread_ownership", None),
        principal_token=principal.access_token,
        visualization_store=getattr(
            request.app.state,
            "visualization_store",
            None,
        ),
    )


@router.get("/v1/chat/{thread_id}/history", response_model=HistoryResponse)
async def chat_history(
    thread_id: str,
    request: Request,
    user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    principal: Annotated[Principal, Depends(get_current_principal)] = None,
) -> HistoryResponse:
    """Retorna uma pagina do historico visivel de uma thread."""
    user_id = user_id_for_request(user_id, principal)
    return await get_thread_history(
        request.app.state.checkpointer,
        thread_id,
        user_id,
        limit,
        before,
        getattr(request.app.state, "visualization_store", None),
    )
