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
    mongodb_database: str = "mongodb-ai-prod"
    groq_api_key: SecretStr | None = None
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_model: str = "openai/gpt-oss-120b"
    nvidia_api_key: SecretStr | None = None
    nvidia_nim_fast_model: str = "nvidia/nemotron-3-nano-30b-a3b"
    nvidia_nim_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30
    llm_total_timeout_seconds: float = 60
    auth_bearer_token: SecretStr | None = None
    auth_jwt_secret: SecretStr | None = None
    auth_jwt_issuer: str | None = None
    auth_jwt_audience: str | None = None
    auth_require_user_jwt: bool = False
    mcp_url: str | None = None
    mcp_access_token: SecretStr | None = None
    mcp_jwt_secret: SecretStr | None = None
    mcp_jwt_issuer_url: str = "https://auth.ouros.local"
    mcp_resource_url: str | None = None
    mcp_user_type: str = "farm_owner"
    mcp_jwt_ttl_seconds: int = 300

    @field_validator(
        "groq_api_key",
        "nvidia_api_key",
        "auth_bearer_token",
        "auth_jwt_secret",
        "mcp_access_token",
        "mcp_jwt_secret",
        mode="before",
    )
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
