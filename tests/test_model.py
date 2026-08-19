import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import SecretStr

from app.agents.model import build_chat_model
from app.core.config import Settings


class ModelTest(unittest.TestCase):
    def test_model_is_disabled_when_all_keys_are_missing(self) -> None:
        self.assertIsNone(build_chat_model(Settings(_env_file=None)))

    def test_groq_can_run_without_nim_fallback(self) -> None:
        groq = Mock()
        config = Settings(_env_file=None, groq_api_key=SecretStr("groq-key"))

        with patch.dict(
            sys.modules,
            {"langchain_groq": SimpleNamespace(ChatGroq=groq)},
        ):
            model = build_chat_model(config)

        self.assertEqual(model, groq.return_value)
        groq.return_value.with_fallbacks.assert_not_called()

    def test_nim_can_run_without_groq(self) -> None:
        nvidia = Mock()
        config = Settings(_env_file=None, nvidia_api_key=SecretStr("nvidia-key"))

        with patch.dict(
            sys.modules,
            {"langchain_openai": SimpleNamespace(ChatOpenAI=nvidia)},
        ):
            model = build_chat_model(config)

        self.assertEqual(model, nvidia.return_value)

    def test_groq_uses_nim_as_fallback(self) -> None:
        groq = Mock()
        nvidia = Mock()
        primary = groq.return_value
        primary.with_fallbacks.return_value = "model"
        config = Settings(
            _env_file=None,
            groq_api_key=SecretStr("groq-key"),
            nvidia_api_key=SecretStr("nvidia-key"),
        )

        with patch.dict(
            sys.modules,
            {
                "langchain_groq": SimpleNamespace(ChatGroq=groq),
                "langchain_openai": SimpleNamespace(ChatOpenAI=nvidia),
            },
        ):
            model = build_chat_model(config)

        self.assertEqual(model, "model")
        groq.assert_called_once_with(
            model_name="openai/gpt-oss-120b",
            api_key="groq-key",
            temperature=0.2,
            timeout=30,
            max_retries=0,
        )
        nvidia.assert_called_once_with(
            model="meta/llama-3.3-70b-instruct",
            api_key="nvidia-key",
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.2,
            timeout=30,
            max_retries=0,
        )
        primary.with_fallbacks.assert_called_once_with([nvidia.return_value])

    def test_blank_secret_values_are_disabled(self) -> None:
        config = Settings(_env_file=None, groq_api_key="   ", nvidia_api_key="\t")

        self.assertIsNone(config.groq_api_key)
        self.assertIsNone(config.nvidia_api_key)
