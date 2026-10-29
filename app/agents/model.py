from functools import lru_cache

from app.agents.llms import FAST_LLM, POWERFUL_LLM
from app.core.config import Settings, settings


def _build_chat_models(config: Settings, profile: str) -> tuple:
    if not config.groq_api_key and not config.nvidia_api_key:
        return ()

    models = []
    if config.groq_api_key:
        from langchain_groq import ChatGroq

        model_name = config.groq_fast_model if profile == FAST_LLM else config.groq_model
        models.append(
            ChatGroq(
                model_name=model_name,
                api_key=config.groq_api_key.get_secret_value(),
                temperature=config.llm_temperature,
                timeout=config.llm_timeout_seconds,
                max_retries=0,
            )
        )
    if config.nvidia_api_key:
        from langchain_openai import ChatOpenAI

        fallback_model = (
            config.nvidia_nim_fast_model
            if profile == FAST_LLM
            else config.nvidia_nim_model
        )
        models.append(
            ChatOpenAI(
                model=fallback_model,
                api_key=config.nvidia_api_key.get_secret_value(),
                base_url=config.nvidia_nim_base_url,
                temperature=config.llm_temperature,
                timeout=config.llm_timeout_seconds,
                max_retries=0,
            )
        )
    return tuple(models)


def _with_fallbacks(models: tuple):
    return models[0] if len(models) == 1 else models[0].with_fallbacks(list(models[1:]))


def build_chat_model(config: Settings, profile: str = POWERFUL_LLM):
    """Monta o perfil selecionado, usando NVIDIA NIM como fallback."""
    models = _build_chat_models(config, profile)
    return _with_fallbacks(models) if models else None


def build_structured_chat_model(
    config: Settings,
    schema,
    profile: str = POWERFUL_LLM,
):
    """Aplica structured output a cada provedor antes de compor os fallbacks."""
    models = tuple(
        model.with_structured_output(schema)
        for model in _build_chat_models(config, profile)
    )
    return _with_fallbacks(models) if models else None


@lru_cache
def get_chat_model(profile: str = POWERFUL_LLM):
    """Retorna o modelo compartilhado do perfil solicitado."""
    return build_chat_model(settings, profile)


@lru_cache
def get_structured_chat_model(schema, profile: str = POWERFUL_LLM):
    """Retorna o modelo estruturado compartilhado para o schema solicitado."""
    return build_structured_chat_model(settings, schema, profile)
