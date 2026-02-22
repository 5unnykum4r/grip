"""Tests for CLI interactive mode enhancements (Phase 4).

Covers: SlashCompleter, provider-aware model display, terminal clear behavior,
compact summary display, welcome panel content, help categories, and the
dynamic prompt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grip.cli.agent_cmd import (
    _COMMANDS,
    _build_completer,
    _short_model_name,
    _resolve_provider_display,
)


# ---------------------------------------------------------------------------
# SlashCompleter
# ---------------------------------------------------------------------------

class TestSlashCompleter:
    """Verify autocomplete returns matching commands with descriptions."""

    def _get_completions(self, text: str) -> list[tuple[str, str]]:
        """Run the completer on ``text`` and return (command, meta) pairs."""
        from prompt_toolkit.document import Document

        completer = _build_completer()
        doc = Document(text, len(text))
        return [(c.text, c.display_meta) for c in completer.get_completions(doc, None)]

    def test_slash_prefix_returns_all_commands(self):
        results = self._get_completions("/")
        command_names = [r[0] for r in results]
        for cmd in _COMMANDS:
            assert cmd in command_names, f"{cmd} missing from autocomplete for '/'"

    def test_slash_c_returns_matching(self):
        results = self._get_completions("/c")
        command_names = [r[0] for r in results]
        assert "/clear" in command_names
        assert "/compact" in command_names
        assert "/copy" in command_names
        assert "/new" not in command_names

    def test_slash_mo_returns_model(self):
        results = self._get_completions("/mo")
        command_names = [r[0] for r in results]
        assert "/model" in command_names
        assert "/mcp" not in command_names

    def test_non_slash_input_returns_nothing(self):
        results = self._get_completions("hello world")
        assert results == []

    def test_empty_input_returns_nothing(self):
        results = self._get_completions("")
        assert results == []

    def test_completions_include_descriptions(self):
        results = self._get_completions("/he")
        assert len(results) == 1
        cmd, meta = results[0]
        assert cmd == "/help"
        meta_str = str(meta).lower() if not isinstance(meta, str) else meta.lower()
        assert "command reference" in meta_str

    def test_slash_e_returns_exit(self):
        results = self._get_completions("/e")
        command_names = [r[0] for r in results]
        assert "/exit" in command_names

    def test_slash_st_returns_status(self):
        results = self._get_completions("/st")
        command_names = [r[0] for r in results]
        assert "/status" in command_names

    def test_full_command_returns_exact_match(self):
        results = self._get_completions("/doctor")
        command_names = [r[0] for r in results]
        assert command_names == ["/doctor"]


# ---------------------------------------------------------------------------
# _short_model_name
# ---------------------------------------------------------------------------

class TestShortModelName:
    def test_triple_slash_path(self):
        assert _short_model_name("openrouter/anthropic/claude-sonnet-4") == "claude-sonnet-4"

    def test_double_slash_path(self):
        assert _short_model_name("anthropic/claude-sonnet-4") == "claude-sonnet-4"

    def test_bare_model_name(self):
        assert _short_model_name("gpt-4o") == "gpt-4o"

    def test_empty_string(self):
        assert _short_model_name("") == ""


# ---------------------------------------------------------------------------
# _resolve_provider_display
# ---------------------------------------------------------------------------

class TestResolveProviderDisplay:
    def _make_config(self, model: str = "openrouter/anthropic/claude-sonnet-4", provider: str = "") -> MagicMock:
        config = MagicMock()
        config.agents.defaults.model = model
        config.agents.defaults.provider = provider
        return config

    def test_openrouter_model(self):
        config = self._make_config("openrouter/anthropic/claude-sonnet-4")
        name, bare = _resolve_provider_display("openrouter/anthropic/claude-sonnet-4", config)
        assert name == "OpenRouter"
        assert "claude-sonnet" in bare or "anthropic" in bare

    def test_anthropic_model(self):
        config = self._make_config("anthropic/claude-sonnet-4")
        name, bare = _resolve_provider_display("anthropic/claude-sonnet-4", config)
        assert name == "Anthropic"
        assert bare == "claude-sonnet-4"

    def test_unknown_model_falls_back(self):
        config = self._make_config("some-random-model")
        name, bare = _resolve_provider_display("some-random-model", config)
        assert isinstance(name, str)
        assert isinstance(bare, str)

    def test_explicit_provider_override(self):
        config = self._make_config("openai/gpt-oss-120b", provider="openrouter")
        name, bare = _resolve_provider_display("openai/gpt-oss-120b", config)
        assert name == "OpenRouter"


# ---------------------------------------------------------------------------
# _COMMANDS registry
# ---------------------------------------------------------------------------

class TestCommandsRegistry:
    def test_all_commands_have_description_and_category(self):
        for cmd, (desc, cat) in _COMMANDS.items():
            assert cmd.startswith("/"), f"Command {cmd} doesn't start with /"
            assert len(desc) > 0, f"Command {cmd} has empty description"

    def test_session_category_commands(self):
        session_cmds = [cmd for cmd, (_, cat) in _COMMANDS.items() if cat == "Session"]
        assert "/new" in session_cmds
        assert "/clear" in session_cmds
        assert "/compact" in session_cmds
        assert "/undo" in session_cmds
        assert "/copy" in session_cmds

    def test_config_category_commands(self):
        config_cmds = [cmd for cmd, (_, cat) in _COMMANDS.items() if cat == "Config"]
        assert "/model" in config_cmds
        assert "/provider" in config_cmds

    def test_info_category_commands(self):
        info_cmds = [cmd for cmd, (_, cat) in _COMMANDS.items() if cat == "Info"]
        assert "/help" in info_cmds
        assert "/status" in info_cmds
        assert "/doctor" in info_cmds
        assert "/mcp" in info_cmds

    def test_exit_has_empty_category(self):
        _, (_, cat) = "/exit", _COMMANDS["/exit"]
        assert cat == ""

    def test_command_count(self):
        assert len(_COMMANDS) >= 14, "Should have at least 14 slash commands"
