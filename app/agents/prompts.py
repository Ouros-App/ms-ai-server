SYSTEM_PROMPT = """Voce e o agente principal deste sistema.
Defina aqui as regras comuns quando o objetivo da IA estiver definido.
"""

ROUTER_PROMPT = """Classifique a mensagem e escolha um agente permitido.
Defina as rotas quando os agentes existirem.
"""

DEFAULT_AGENT_RESPONSE = "A IA ainda nao foi configurada. Defina os agentes, prompts e tools do projeto."

AGENT_PROMPTS: dict[str, str] = {}
