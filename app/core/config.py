from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracao tipada carregada do ambiente."""
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    project_name: str = "AI Server"
    description: str = "API de orquestracao de IA."
    version: str = "0.1.0"
    app_port: int = 8000
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "ai_server"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    nvidia_api_key: SecretStr | None = None
    nvidia_nim_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30
    llm_total_timeout_seconds: float = 20
    auth_bearer_token: SecretStr | None = None

    @field_validator("groq_api_key", "nvidia_api_key", "auth_bearer_token", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value):
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Retorna a configuracao compartilhada da aplicacao."""
    return Settings()


settings = get_settings()
