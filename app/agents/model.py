from functools import lru_cache

from app.core.config import Settings, settings


def build_chat_model(config: Settings):
    """Monta Groq com NVIDIA NIM como fallback quando as chaves existem."""
    if not config.groq_api_key or not config.nvidia_api_key:
        return None

    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    primary = ChatGroq(
        model_name=config.groq_model,
        api_key=config.groq_api_key.get_secret_value(),
        temperature=config.llm_temperature,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    fallback = ChatOpenAI(
        model=config.nvidia_nim_model,
        api_key=config.nvidia_api_key.get_secret_value(),
        base_url=config.nvidia_nim_base_url,
        temperature=config.llm_temperature,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    return primary.with_fallbacks([fallback])


@lru_cache
def get_chat_model():
    """Retorna o modelo compartilhado configurado para o processo."""
    return build_chat_model(settings)
