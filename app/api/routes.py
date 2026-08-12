from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import HealthResponse, MessageResponse
from app.services.chat import invoke_graph

router = APIRouter()


@router.get("/", response_model=MessageResponse)
async def read_root() -> MessageResponse:
    return MessageResponse(message="AI Server is running")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/v1/chat")
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return await invoke_graph(request.app.state.graph, payload)
