import os

from dotenv import load_dotenv
from infisical_sdk import InfisicalSDKClient


def load_infisical_secrets() -> None:
    load_dotenv()
    client = InfisicalSDKClient(
        host=os.getenv("INFISICAL_HOST", "https://app.infisical.com"),
        token=os.environ["INFISICAL_TOKEN"],
    )
    response = client.secrets.list_secrets(
        project_id=os.environ["INFISICAL_PROJECT_ID"],
        environment_slug=os.getenv("INFISICAL_ENV", "prod"),
        secret_path=os.environ["INFISICAL_PATH"],
        view_secret_value=True,
    )
    for secret in response.secrets:
        os.environ[secret.secretKey] = secret.secretValue
