FAST_LLM = "fast"
POWERFUL_LLM = "powerful"

AGENT_LLM_PROFILE = {
    "router": FAST_LLM,
    "faq": FAST_LLM,
    "support": FAST_LLM,
    "fallback": FAST_LLM,
    "sustainability": POWERFUL_LLM,
    "ranking": POWERFUL_LLM,
    "default": POWERFUL_LLM,
    "guardrail": FAST_LLM,
}


def profile_for(agent_name: str) -> str:
    return AGENT_LLM_PROFILE.get(agent_name, POWERFUL_LLM)
