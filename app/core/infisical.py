import os

from dotenv import load_dotenv
from infisical_sdk import InfisicalSDKClient


def load_infisical_secrets() -> None:
    """Carrega secrets do Infisical quando o runtime foi configurado para isso."""
    load_dotenv()
    token = os.getenv("INFISICAL_TOKEN")
    project_id = os.getenv("INFISICAL_PROJECT_ID")
    environment = os.getenv("INFISICAL_ENV")
    secret_path = os.getenv("INFISICAL_PATH")
    if not any((token, project_id, environment, secret_path)):
        return
    missing = [
        name
        for name, value in (
            ("INFISICAL_TOKEN", token),
            ("INFISICAL_PROJECT_ID", project_id),
            ("INFISICAL_ENV", environment),
            ("INFISICAL_PATH", secret_path),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Configuracao incompleta do Infisical: {', '.join(missing)}")
    if environment not in {"prod", "dev"}:
        raise RuntimeError("INFISICAL_ENV deve ser 'prod' ou 'dev'")
    client = InfisicalSDKClient(
        host=os.getenv("INFISICAL_HOST", "https://app.infisical.com"),
        token=token,
    )
    response = client.secrets.list_secrets(
        project_id=project_id,
        environment_slug=os.getenv("INFISICAL_ENV", "prod"),
        secret_path=secret_path,
        view_secret_value=True,
    )
    for secret in response.secrets:
        os.environ[secret.secretKey] = secret.secretValue
