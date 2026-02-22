"""Tests for the configuration system."""

from __future__ import annotations

import json

from grip.config.schema import (
    AgentDefaults,
    AgentProfile,
    APIConfig,
    GatewayConfig,
    GripConfig,
    ProviderEntry,
)


def test_default_config_loads():
    """AgentDefaults should have sensible defaults when constructed directly."""
    defaults = AgentDefaults()
    assert defaults.model == "openrouter/anthropic/claude-sonnet-4"
    assert defaults.max_tokens == 8192
    assert defaults.temperature == 0.7


def test_api_config_defaults():
    """APIConfig should have secure defaults."""
    api = APIConfig()
    assert api.auth_token == ""
    assert api.rate_limit_per_minute == 60
    assert api.rate_limit_per_minute_per_ip == 30
    assert api.enable_tool_execute is False
    assert api.max_request_body_bytes == 1_048_576


def test_gateway_includes_api():
    """GatewayConfig should nest APIConfig."""
    gw = GatewayConfig()
    assert gw.host == "127.0.0.1"
    assert gw.port == 18800
    assert isinstance(gw.api, APIConfig)


def test_agent_profile_empty_inherits():
    """Empty AgentProfile fields signal inheritance from defaults."""
    profile = AgentProfile()
    assert profile.model == ""
    assert profile.tools_allowed == []
    assert profile.tools_denied == []


def test_agent_profiles_in_config():
    """Profiles dict should be accessible from agents config."""
    config = GripConfig(
        agents={
            "defaults": {},
            "profiles": {
                "researcher": {
                    "model": "openai/gpt-4o",
                    "tools_allowed": ["web_search", "web_fetch"],
                },
            },
        }
    )
    assert "researcher" in config.agents.profiles
    assert config.agents.profiles["researcher"].model == "openai/gpt-4o"


def test_workspace_path_expansion():
    """Workspace path with ~ should be expandable."""
    defaults = AgentDefaults()
    expanded = defaults.workspace.expanduser()
    assert "~" not in str(expanded)


def test_save_config_strips_empty_providers(tmp_path):
    """save_config should not write provider entries with no api_key and no default_model."""
    from grip.config import save_config

    config = GripConfig(
        providers={
            "openrouter": ProviderEntry(api_key="real-key", default_model="gpt-4o"),
            "lmstudio": ProviderEntry(),
        },
    )
    path = tmp_path / "config.json"
    save_config(config, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "openrouter" in saved["providers"]
    assert "lmstudio" not in saved["providers"]


def test_save_config_keeps_provider_with_key(tmp_path):
    """Providers with api_key set should not be stripped."""
    from grip.config import save_config

    config = GripConfig(
        providers={
            "anthropic": ProviderEntry(api_key="sk-test"),
        },
    )
    path = tmp_path / "config.json"
    save_config(config, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "anthropic" in saved["providers"]


def test_save_config_keeps_provider_with_default_model(tmp_path):
    """Providers with default_model set should not be stripped even without api_key."""
    from grip.config import save_config

    config = GripConfig(
        providers={
            "ollama": ProviderEntry(default_model="llama3.2"),
        },
    )
    path = tmp_path / "config.json"
    save_config(config, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "ollama" in saved["providers"]
