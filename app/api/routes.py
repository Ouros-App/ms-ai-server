from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.core.auth import Principal, get_current_principal
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import HealthResponse, MessageResponse
from app.services.chat import invoke_graph

router = APIRouter()


@router.get("/", response_model=MessageResponse)
async def read_root() -> MessageResponse:
    """Informa que a API esta disponivel."""
    return MessageResponse(message="AI Server is running")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Retorna o estado de saude da API."""
    return HealthResponse(status="ok")


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    _principal: Annotated[Principal, Depends(get_current_principal)],
) -> ChatResponse:
    """Processa uma mensagem dentro de uma thread persistente."""
    return await invoke_graph(request.app.state.graph, payload, payload.user_id)
