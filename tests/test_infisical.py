import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.infisical import load_infisical_secrets


class InfisicalTest(unittest.TestCase):
    def test_skips_when_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("app.core.infisical.load_dotenv"):
            load_infisical_secrets()

    def test_rejects_incomplete_configuration(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"INFISICAL_TOKEN": "token"},
                clear=True,
            ),
            patch("app.core.infisical.load_dotenv"),
            self.assertRaisesRegex(
                RuntimeError,
                "Configuracao incompleta do Infisical",
            ),
        ):
            load_infisical_secrets()

    def test_rejects_unknown_environment(self) -> None:
        env = {
            "INFISICAL_TOKEN": "token",
            "INFISICAL_PROJECT_ID": "project",
            "INFISICAL_ENV": "staging",
            "INFISICAL_PATH": "/service",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(
            RuntimeError, "prod.*dev"
        ):
            load_infisical_secrets()

    def test_loads_secrets(self) -> None:
        env = {
            "INFISICAL_TOKEN": "token",
            "INFISICAL_PROJECT_ID": "project",
            "INFISICAL_ENV": "dev",
            "INFISICAL_PATH": "/service",
            "API_KEY": "local",
        }
        response = SimpleNamespace(
            secrets=[SimpleNamespace(secretKey="API_KEY", secretValue="loaded")]
        )
        with patch.dict(os.environ, env, clear=True), patch(
            "app.core.infisical.InfisicalSDKClient"
        ) as client:
            client.return_value.secrets.list_secrets.return_value = response
            load_infisical_secrets()
            self.assertEqual(os.environ["API_KEY"], "loaded")

        client.assert_called_once_with(
            host="https://app.infisical.com",
            token="token",
        )
        client.return_value.secrets.list_secrets.assert_called_once_with(
            project_id="project",
            environment_slug="dev",
            secret_path="/service",
            view_secret_value=True,
        )

    def test_config_bootstraps_infisical_before_settings(self) -> None:
        env = {
            "INFISICAL_TOKEN": "token",
            "INFISICAL_PROJECT_ID": "project",
            "INFISICAL_ENV": "dev",
            "INFISICAL_PATH": "/ms-ai-server",
        }
        response = SimpleNamespace(
            secrets=[
                SimpleNamespace(
                    secretKey="AUTH_BEARER_TOKEN",
                    secretValue="from-infisical",
                )
            ]
        )

        with (
            patch.dict(os.environ, env, clear=True),
            patch("app.core.infisical.load_dotenv"),
            patch("app.core.infisical.InfisicalSDKClient") as client,
        ):
            client.return_value.secrets.list_secrets.return_value = response
            from app.core import config

            importlib.reload(config)
            self.assertEqual(
                config.settings.auth_bearer_token.get_secret_value(),
                "from-infisical",
            )

        importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
