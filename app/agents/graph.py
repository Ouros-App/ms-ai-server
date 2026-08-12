from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.agents.model import get_chat_model
from app.agents.prompts import DEFAULT_AGENT_RESPONSE, SYSTEM_PROMPT


class AgentState(MessagesState):
    user_id: str
    route: str
    agents: list[str]


def route_request(state: AgentState) -> dict:
    """Seleciona o agente padrao enquanto nao ha agentes especializados."""
    _ = state
    return {"route": "default", "agents": ["router"]}


async def default_agent(state: AgentState) -> dict:
    """Gera uma resposta do modelo ou o placeholder configurado."""
    model = get_chat_model()
    if model:
        response = await model.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            *state["messages"],
        ])
        message = response
    else:
        message = AIMessage(content=DEFAULT_AGENT_RESPONSE)

    return {
        "agents": [*state["agents"], "default"],
        "messages": [message],
    }


AGENTS = {"default": default_agent}


def choose_agent(state: AgentState) -> str:
    """Converte a rota em um agente registrado com fallback seguro."""
    return state["route"] if state["route"] in AGENTS else "default"


def build_graph(checkpointer):
    """Compila o fluxo roteador-agente usando o checkpoint recebido."""
    graph = StateGraph(AgentState)
    graph.add_node("router", route_request)
    for name, node in AGENTS.items():
        graph.add_node(name, node)
        graph.add_edge(name, END)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", choose_agent, {name: name for name in AGENTS})
    return graph.compile(checkpointer=checkpointer)
