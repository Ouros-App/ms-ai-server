import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import SecretStr

from app.agents.model import build_chat_model, build_structured_chat_model
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

    def test_fast_profile_uses_fast_groq_model(self) -> None:
        groq = Mock()
        config = Settings(_env_file=None, groq_api_key=SecretStr("groq-key"))

        with patch.dict(
            sys.modules,
            {"langchain_groq": SimpleNamespace(ChatGroq=groq)},
        ):
            build_chat_model(config, "fast")

        self.assertEqual(groq.call_args.kwargs["model_name"], "openai/gpt-oss-20b")

    def test_nim_can_run_without_groq(self) -> None:
        nvidia = Mock()
        config = Settings(_env_file=None, nvidia_api_key=SecretStr("nvidia-key"))

        with patch.dict(
            sys.modules,
            {"langchain_openai": SimpleNamespace(ChatOpenAI=nvidia)},
        ):
            model = build_chat_model(config)

        self.assertEqual(model, nvidia.return_value)

    def test_fast_profile_uses_fast_nim_model(self) -> None:
        nvidia = Mock()
        config = Settings(_env_file=None, nvidia_api_key=SecretStr("nvidia-key"))

        with patch.dict(
            sys.modules,
            {"langchain_openai": SimpleNamespace(ChatOpenAI=nvidia)},
        ):
            build_chat_model(config, "fast")

        self.assertEqual(
            nvidia.call_args.kwargs["model"],
            "nvidia/nemotron-3-nano-30b-a3b",
        )

    def test_openai_is_primary_with_groq_then_nim_fallbacks(self) -> None:
        groq = Mock()
        openai_primary = Mock()
        groq_fallback = Mock()
        nvidia_fallback = Mock()
        openai = Mock(side_effect=[openai_primary, nvidia_fallback])
        groq.return_value = groq_fallback
        openai_primary.with_fallbacks.return_value = "model"
        config = Settings(
            _env_file=None,
            openai_api_key=SecretStr("openai-key"),
            groq_api_key=SecretStr("groq-key"),
            nvidia_api_key=SecretStr("nvidia-key"),
        )

        with patch.dict(
            sys.modules,
            {
                "langchain_groq": SimpleNamespace(ChatGroq=groq),
                "langchain_openai": SimpleNamespace(ChatOpenAI=openai),
            },
        ):
            model = build_chat_model(config)

        self.assertEqual(model, "model")
        openai_kwargs = openai.call_args_list[0].kwargs
        self.assertEqual(openai_kwargs["model"], "gpt-6-luna")
        self.assertEqual(openai_kwargs["reasoning_effort"], "none")
        self.assertEqual(openai_kwargs["max_completion_tokens"], 1024)
        self.assertEqual(openai_kwargs["max_retries"], 0)
        self.assertEqual(openai_kwargs["callbacks"][0].provider, "openai")
        self.assertFalse(openai_kwargs["callbacks"][0].is_fallback)
        groq_kwargs = groq.call_args.kwargs
        self.assertEqual(groq_kwargs["callbacks"][0].provider, "groq")
        self.assertTrue(groq_kwargs["callbacks"][0].is_fallback)
        self.assertEqual(
            openai_primary.with_fallbacks.call_args.args[0],
            [groq_fallback, nvidia_fallback],
        )

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
        groq_kwargs = groq.call_args.kwargs
        self.assertEqual(groq_kwargs["model_name"], "openai/gpt-oss-120b")
        self.assertEqual(groq_kwargs["api_key"], "groq-key")
        self.assertEqual(groq_kwargs["temperature"], 0.2)
        self.assertEqual(groq_kwargs["timeout"], 30)
        self.assertEqual(groq_kwargs["max_retries"], 0)
        self.assertFalse(groq_kwargs["callbacks"][0].is_fallback)
        nvidia_kwargs = nvidia.call_args.kwargs
        self.assertEqual(nvidia_kwargs["model"], "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(nvidia_kwargs["api_key"], "nvidia-key")
        self.assertEqual(nvidia_kwargs["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(nvidia_kwargs["temperature"], 0.2)
        self.assertEqual(nvidia_kwargs["timeout"], 30)
        self.assertEqual(nvidia_kwargs["max_retries"], 0)
        self.assertTrue(nvidia_kwargs["callbacks"][0].is_fallback)
        primary.with_fallbacks.assert_called_once_with([nvidia.return_value])


    def test_structured_output_wraps_each_provider_before_fallback(self) -> None:
        groq = Mock()
        nvidia = Mock()
        schema = object()
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
            model = build_structured_chat_model(config, schema)

        groq.return_value.with_structured_output.assert_called_once_with(schema)
        nvidia.return_value.with_structured_output.assert_called_once_with(schema)
        structured_primary = groq.return_value.with_structured_output.return_value
        structured_fallback = nvidia.return_value.with_structured_output.return_value
        structured_primary.with_fallbacks.assert_called_once_with([structured_fallback])
        self.assertEqual(model, structured_primary.with_fallbacks.return_value)

    def test_blank_secret_values_are_disabled(self) -> None:
        config = Settings(
            _env_file=None,
            openai_api_key=" ",
            groq_api_key="   ",
            nvidia_api_key="\t",
        )

        self.assertIsNone(config.openai_api_key)
        self.assertIsNone(config.groq_api_key)
        self.assertIsNone(config.nvidia_api_key)
