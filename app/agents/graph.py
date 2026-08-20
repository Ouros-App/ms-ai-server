import json
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.agents.guardrails import (
    OUT_OF_SCOPE_REFUSAL,
    guard_input,
    review_output,
)
from app.agents.llms import profile_for
from app.agents.model import get_chat_model
from app.agents.prompts import (
    AGENT_PROMPTS,
    DEFAULT_AGENT_RESPONSE,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
)
from app.agents.tools import build_memory_tools
from app.core.config import settings

logger = logging.getLogger(__name__)
ROUTES = frozenset(AGENT_PROMPTS)


class AgentState(MessagesState):
    user_id: str
    route: str
    agents: list[str]
    tools: list[str]
    input_guardrail: dict[str, object]


def _extract_route(response: object) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return "fallback"

    try:
        payload = json.loads(content.strip())
    except json.JSONDecodeError:
        return "fallback"

    route = payload.get("route") if isinstance(payload, dict) else None
    normalized_route = route.lower() if isinstance(route, str) else ""
    return normalized_route if normalized_route in ROUTES else "fallback"


async def route_request(state: AgentState) -> dict:
    """Preserva uma rota explicita ou escolhe o agente com o modelo."""
    input_guardrail = state.get("input_guardrail")
    if input_guardrail and not input_guardrail.get("allowed", True):
        route = "default"
    else:
        requested_route = state.get("route")
        if requested_route:
            route = requested_route
        else:
            model = get_chat_model(profile_for("router"))
            if model is None:
                route = "default"
            else:
                try:
                    response = await model.ainvoke([
                        {"role": "system", "content": ROUTER_PROMPT},
                        *state.get("messages", [])[-6:],
                    ])
                    route = _extract_route(response)
                except Exception:
                    logger.exception("agent_router_failed")
                    route = "fallback"
                logger.info("agent_route_selected route=%s", route)

    return {
        "route": route,
        "agents": ["router"],
        "tools": [],
    }


async def default_agent(state: AgentState) -> dict:
    """Gera uma resposta generica do modelo ou o placeholder configurado."""
    return await _run_agent(state, SYSTEM_PROMPT, "default")


async def _run_agent(state: AgentState, prompt: str, agent_name: str) -> dict:
    """Executa um agente com guardrails, memoria e revisao de saida."""
    latest_message = state["messages"][-1] if state["messages"] else None
    user_text = getattr(latest_message, "content", "")
    used_tools: list[str] = []
    input_guardrail = state.get("input_guardrail")
    if input_guardrail is None and isinstance(user_text, str):
        decision = await guard_input(user_text, has_history=len(state["messages"]) > 1)
        input_guardrail = decision.as_state()
    if input_guardrail and not input_guardrail["allowed"]:
        content = input_guardrail["message"]
    elif not isinstance(user_text, str):
        content = OUT_OF_SCOPE_REFUSAL
    else:
        model = get_chat_model(profile_for(agent_name))
        if model:
            response, used_tools = await _invoke_model(
                model,
                [
                    {"role": "system", "content": prompt},
                    *state["messages"],
                ],
                state.get("memory_store"),
                state["user_id"],
            )
            content = getattr(response, "content", response)
        else:
            content = DEFAULT_AGENT_RESPONSE

    if used_tools:
        logger.info("agent_tools_used agent=%s tools=%s", agent_name, used_tools)

    sensitive_token = (
        settings.auth_bearer_token.get_secret_value() if settings.auth_bearer_token else ""
    )
    message = AIMessage(content=await review_output(content, sensitive_token))

    return {
        "agents": [*state["agents"], agent_name],
        "tools": list(dict.fromkeys([*state.get("tools", []), *used_tools])),
        "messages": [message],
    }


async def _invoke_model(model, messages: list, memory_store, user_id: str):
    """Executa o modelo e atende chamadas das tools de memoria."""
    if memory_store is None:
        return await model.ainvoke(messages), []

    tools = build_memory_tools(memory_store, user_id)
    model_with_tools = model.bind_tools(tools)
    tool_map = {memory_tool.name: memory_tool for memory_tool in tools}
    conversation = list(messages)
    used_tools: list[str] = []

    for _ in range(3):
        response = await model_with_tools.ainvoke(conversation)
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            return response, used_tools

        conversation.append(response)
        for call in tool_calls:
            memory_tool = tool_map.get(call["name"])
            if memory_tool is None:
                continue
            if memory_tool.name not in used_tools:
                used_tools.append(memory_tool.name)
            result = await memory_tool.ainvoke(call.get("args", {}))
            conversation.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call.get("id", f"tool-call-{len(conversation)}"),
                ),
            )

    return await model.ainvoke(conversation), used_tools


def _build_prompt_agent(name: str, prompt: str):
    async def agent(state: AgentState) -> dict:
        return await _run_agent(state, prompt, name)

    return agent


AGENTS = {
    "default": default_agent,
    **{
        name: _build_prompt_agent(name, prompt)
        for name, prompt in AGENT_PROMPTS.items()
    },
}


def choose_agent(state: AgentState, agents: dict) -> str:
    """Converte a rota em um agente registrado com fallback seguro."""
    return state["route"] if state["route"] in agents else "default"


def build_graph(checkpointer, agents: dict | None = None, memory_store=None):
    """Compila o fluxo usando o registro de agentes recebido."""
    agents = AGENTS if agents is None else agents
    if "default" not in agents:
        raise ValueError("O registro de agentes precisa conter o agente default.")

    graph = StateGraph(AgentState)
    graph.add_node("router", route_request)
    for name, node in agents.items():
        if memory_store is not None:
            async def node_with_memory(state, node=node):
                state = {**state, "memory_store": memory_store}
                return await node(state)

            node = node_with_memory
        graph.add_node(name, node)
        graph.add_edge(name, END)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: choose_agent(state, agents),
        {name: name for name in agents},
    )
    return graph.compile(checkpointer=checkpointer)
