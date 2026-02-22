"""Tests for the onboarding wizard flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from grip.config.schema import GripConfig
from grip.providers.registry import ProviderRegistry
from grip.providers.types import LLMResponse, TokenUsage, ToolCall


class TestProviderChoiceConstants:
    """Verify the provider ordering constants match the registry."""

    def test_cloud_providers_match_expected_order(self):
        from grip.cli.onboard import _CLOUD_PROVIDERS

        expected = [
            "openrouter",
            "anthropic",
            "openai",
            "deepseek",
            "groq",
            "gemini",
            "qwen",
            "minimax",
            "moonshot",
            "ollama_cloud",
        ]
        assert expected == _CLOUD_PROVIDERS

    def test_all_cloud_providers_exist_in_registry(self):
        from grip.cli.onboard import _CLOUD_PROVIDERS

        for name in _CLOUD_PROVIDERS:
            spec = ProviderRegistry.get_spec(name)
            assert spec is not None, f"Provider '{name}' not found in registry"

    def test_local_providers_exist_in_registry(self):
        from grip.cli.onboard import _LOCAL_SUBMENU

        for name in _LOCAL_SUBMENU:
            spec = ProviderRegistry.get_spec(name)
            assert spec is not None, f"Local provider '{name}' not found in registry"

    def test_llamacpp_in_registry(self):
        spec = ProviderRegistry.get_spec("llamacpp")
        assert spec is not None
        assert spec.display_name == "Llama.cpp (Local)"
        assert spec.api_base == "http://localhost:8080/v1"


class TestBuildProviderChoices:
    """Test the InquirerPy choice builder functions."""

    def test_provider_choices_count(self):
        from grip.cli.onboard import _build_provider_choices

        choices = _build_provider_choices()
        # Claude SDK + 10 cloud providers + "Ollama (Local)" + "Other Models"
        assert len(choices) == 13

    def test_provider_choices_end_with_local_options(self):
        from grip.cli.onboard import _build_provider_choices

        choices = _build_provider_choices()
        assert choices[-2].value == "_local_ollama"
        assert choices[-1].value == "_local_other"

    def test_provider_choices_start_with_openrouter(self):
        from grip.cli.onboard import _build_provider_choices

        choices = _build_provider_choices()
        assert choices[0].value == "_claude_sdk"

    def test_local_choices_count(self):
        from grip.cli.onboard import _build_local_choices

        choices = _build_local_choices()
        assert len(choices) == 4

    def test_local_choices_includes_custom(self):
        from grip.cli.onboard import _build_local_choices

        choices = _build_local_choices()
        values = [c.value for c in choices]
        assert "_custom_openai" in values
        assert "ollama" in values
        assert "llamacpp" in values
        assert "lmstudio" in values


class TestAutoTestConnection:
    """Test the auto-test connection function."""

    @patch("grip.cli.onboard.create_provider")
    @patch("grip.cli.onboard.asyncio")
    def test_success_with_tool_calls(self, mock_asyncio, mock_create_provider):
        from grip.cli.onboard import _auto_test_connection

        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        chat_response = LLMResponse(content="grip ready!", usage=TokenUsage(), raw={})
        tool_response = LLMResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function_name="get_current_time",
                    arguments={},
                )
            ],
            usage=TokenUsage(),
            raw={},
        )
        mock_asyncio.run.side_effect = [chat_response, tool_response]

        config = GripConfig()
        result = _auto_test_connection(config, "openrouter/openai/gpt-4o")
        assert result is True

    @patch("grip.cli.onboard.create_provider")
    @patch("grip.cli.onboard.asyncio")
    def test_success_without_tool_calls_still_passes(self, mock_asyncio, mock_create_provider):
        """Chat succeeds but model doesn't use tools — should still return True."""
        from grip.cli.onboard import _auto_test_connection

        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        chat_response = LLMResponse(content="grip ready!", usage=TokenUsage(), raw={})
        no_tool_response = LLMResponse(content="It is 3pm.", usage=TokenUsage(), raw={})
        mock_asyncio.run.side_effect = [chat_response, no_tool_response]

        config = GripConfig()
        result = _auto_test_connection(config, "openrouter/openai/gpt-4o")
        assert result is True

    @patch("grip.cli.onboard.create_provider")
    def test_connection_failure_returns_false(self, mock_create_provider):
        from grip.cli.onboard import _auto_test_connection

        mock_create_provider.side_effect = Exception("Connection refused")

        config = GripConfig()
        result = _auto_test_connection(config, "openrouter/openai/gpt-4o")
        assert result is False


class TestSelectProvider:
    """Test provider selection logic with mocked InquirerPy."""

    @patch("grip.cli.onboard.inquirer")
    def test_cloud_provider_selection(self, mock_inquirer):
        from grip.cli.onboard import _select_provider

        mock_prompt = MagicMock()
        mock_prompt.execute.return_value = "anthropic"
        mock_inquirer.select.return_value = mock_prompt

        name, is_custom = _select_provider()
        assert name == "anthropic"
        assert is_custom is False

    @patch("grip.cli.onboard.inquirer")
    def test_ollama_shortcut(self, mock_inquirer):
        from grip.cli.onboard import _select_provider

        mock_prompt = MagicMock()
        mock_prompt.execute.return_value = "_local_ollama"
        mock_inquirer.select.return_value = mock_prompt

        name, is_custom = _select_provider()
        assert name == "ollama"
        assert is_custom is False

    @patch("grip.cli.onboard.inquirer")
    def test_local_other_custom_openai(self, mock_inquirer):
        from grip.cli.onboard import _select_provider

        # First call returns _local_other, second returns _custom_openai
        mock_prompt1 = MagicMock()
        mock_prompt1.execute.return_value = "_local_other"
        mock_prompt2 = MagicMock()
        mock_prompt2.execute.return_value = "_custom_openai"
        mock_inquirer.select.side_effect = [mock_prompt1, mock_prompt2]

        name, is_custom = _select_provider()
        assert name == ""
        assert is_custom is True

    @patch("grip.cli.onboard.inquirer")
    def test_local_other_llamacpp(self, mock_inquirer):
        from grip.cli.onboard import _select_provider

        mock_prompt1 = MagicMock()
        mock_prompt1.execute.return_value = "_local_other"
        mock_prompt2 = MagicMock()
        mock_prompt2.execute.return_value = "llamacpp"
        mock_inquirer.select.side_effect = [mock_prompt1, mock_prompt2]

        name, is_custom = _select_provider()
        assert name == "llamacpp"
        assert is_custom is False
