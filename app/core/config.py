from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.infisical import load_infisical_secrets

load_infisical_secrets()


class Settings(BaseSettings):
    """Configuracao tipada carregada do ambiente."""
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    project_name: str = "AI Server"
    description: str = "API de orquestracao de IA."
    version: str = "0.1.0"
    app_port: int = 8000
    metrics_token: SecretStr | None = None
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
    auth_jwt_issuer: str = "https://ouros-keycloak.discloud.app/realms/ouros"
    auth_jwt_audience: str = "ms-ai-server"
    auth_jwks_url: str | None = None
    mcp_url: str | None = "https://ms-midas-mcp.discloud.app/mcp/"
    mcp_resource_url: str | None = None
    mcp_tools_cache_ttl_seconds: int = 300
    mcp_tool_timeout_seconds: float = Field(10.0, gt=0, le=30)
    mcp_keycloak_token_exchange_url: str | None = None
    mcp_keycloak_token_exchange_client_id: str = "ms-ai-server-mcp-exchange"
    mcp_keycloak_token_exchange_client_secret: SecretStr | None = None
    mcp_keycloak_token_exchange_audience: str = "ms-mcp-server-ouros-knowledge"
    mcp_keycloak_token_exchange_timeout_seconds: float = Field(8.0, gt=0)
    debug_ui_enabled: bool = False
    debug_ui_keycloak_token_url: str | None = None
    debug_ui_keycloak_client_id: str = "ms-ai-server-debug"
    debug_ui_keycloak_client_secret: SecretStr | None = None
    debug_ui_cookie_secure: bool = True
    debug_ui_request_timeout_seconds: float = 8.0

    @property
    def keycloak_token_url(self) -> str:
        return (
            f"{self.auth_jwt_issuer.rstrip('/')}"
            "/protocol/openid-connect/token"
        )

    @property
    def effective_mcp_token_exchange_url(self) -> str:
        return self.mcp_keycloak_token_exchange_url or self.keycloak_token_url

    @property
    def effective_debug_ui_token_url(self) -> str:
        return self.debug_ui_keycloak_token_url or self.keycloak_token_url

    @model_validator(mode="after")
    def validate_keycloak_jwt_config(self) -> "Settings":
        if not self.auth_jwt_issuer.strip() or not self.auth_jwt_audience.strip():
            raise ValueError(
                "AUTH_JWT_ISSUER e AUTH_JWT_AUDIENCE são obrigatórios"
            )
        if self.mcp_keycloak_token_exchange_client_secret is not None:
            if (
                self.mcp_keycloak_token_exchange_url is not None
                and not self.mcp_keycloak_token_exchange_url.strip()
            ):
                raise ValueError(
                    "MCP_KEYCLOAK_TOKEN_EXCHANGE_URL não pode ser vazio"
                )
            if not self.mcp_keycloak_token_exchange_client_id.strip():
                raise ValueError(
                    "MCP_KEYCLOAK_TOKEN_EXCHANGE_CLIENT_ID é obrigatório quando o exchange está configurado"
                )
            if not self.mcp_keycloak_token_exchange_audience.strip():
                raise ValueError(
                    "MCP_KEYCLOAK_TOKEN_EXCHANGE_AUDIENCE é obrigatório quando o exchange está configurado"
                )
        if self.debug_ui_enabled:
            if (
                self.debug_ui_keycloak_token_url is not None
                and not self.debug_ui_keycloak_token_url.strip()
            ):
                raise ValueError(
                    "DEBUG_UI_KEYCLOAK_TOKEN_URL não pode ser vazio"
                )
            if not self.debug_ui_keycloak_client_id.strip():
                raise ValueError(
                    "DEBUG_UI_KEYCLOAK_CLIENT_ID é obrigatório quando DEBUG_UI_ENABLED=true"
                )
            if self.debug_ui_keycloak_client_secret is None:
                raise ValueError(
                    "DEBUG_UI_KEYCLOAK_CLIENT_SECRET é obrigatório quando DEBUG_UI_ENABLED=true"
                )
        return self

    @field_validator(
        "metrics_token",
        "groq_api_key",
        "nvidia_api_key",
        "mcp_keycloak_token_exchange_client_secret",
        "debug_ui_keycloak_client_secret",
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
