"""Tests for Discord channel command handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from grip.bus.events import InboundMessage
from grip.channels.discord import _DISCORD_COMMANDS
from grip.config.schema import ChannelEntry


def _make_config(**overrides) -> ChannelEntry:
    defaults = {"enabled": True, "token": "test-discord-token", "allow_from": []}
    defaults.update(overrides)
    return ChannelEntry(**defaults)


def _make_discord_message(
    content: str,
    author_id: int = 12345,
    channel_id: int = 99,
    guild_id: int = 1,
    message_id: int = 555,
):
    """Build a mock discord.Message with the given content."""
    author = SimpleNamespace(id=author_id, bot=False)
    guild = SimpleNamespace(id=guild_id)
    channel = SimpleNamespace(id=channel_id)
    return SimpleNamespace(
        content=content,
        author=author,
        guild=guild,
        channel=channel,
        id=message_id,
    )


class TestDiscordCommandParsing:
    def test_discord_commands_dict_has_expected_entries(self):
        expected = {"help", "new", "status", "model", "clear", "compact", "version"}
        assert set(_DISCORD_COMMANDS.keys()) == expected

    def test_command_descriptions_are_strings(self):
        for _cmd, desc in _DISCORD_COMMANDS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestDiscordCommandHandling:
    """Test command parsing within the on_message handler logic.

    Since on_message is a nested closure inside start(), we test the command
    parsing logic by simulating what on_message does with a message.
    """

    @pytest.mark.asyncio
    async def test_help_command_detected_with_slash(self):
        text = "/help"
        assert text.startswith(("/", "!"))
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command == "help"
        assert command in _DISCORD_COMMANDS

    @pytest.mark.asyncio
    async def test_help_command_detected_with_bang(self):
        text = "!help"
        assert text.startswith(("/", "!"))
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command == "help"
        assert command in _DISCORD_COMMANDS

    @pytest.mark.asyncio
    async def test_model_command_extracts_arg(self):
        text = "/model gpt-4o"
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        assert command == "model"
        assert arg == "gpt-4o"
        assert command in _DISCORD_COMMANDS

    @pytest.mark.asyncio
    async def test_unknown_command_not_in_dict(self):
        text = "/unknown"
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command not in _DISCORD_COMMANDS

    @pytest.mark.asyncio
    async def test_regular_message_not_treated_as_command(self):
        text = "Hello, how are you?"
        assert not text.startswith(("/", "!"))

    @pytest.mark.asyncio
    async def test_command_message_creates_correct_inbound(self):
        """Simulate the InboundMessage that on_message creates for a command."""
        msg = _make_discord_message("/new")
        text = msg.content.strip()
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        cmd_msg = InboundMessage(
            channel="discord",
            chat_id=str(msg.channel.id),
            user_id=str(msg.author.id),
            text=f"/{command} {arg}".strip(),
            metadata={
                "message_id": str(msg.id),
                "guild_id": str(msg.guild.id),
                "command": command,
                "arg": arg,
            },
        )
        assert cmd_msg.channel == "discord"
        assert cmd_msg.text == "/new"
        assert cmd_msg.metadata["command"] == "new"
        assert cmd_msg.metadata["arg"] == ""
        assert cmd_msg.chat_id == "99"
        assert cmd_msg.user_id == "12345"

    @pytest.mark.asyncio
    async def test_command_with_arg_creates_correct_inbound(self):
        msg = _make_discord_message("!model claude-3.5-sonnet")
        text = msg.content.strip()
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        cmd_msg = InboundMessage(
            channel="discord",
            chat_id=str(msg.channel.id),
            user_id=str(msg.author.id),
            text=f"/{command} {arg}".strip(),
            metadata={
                "message_id": str(msg.id),
                "guild_id": str(msg.guild.id),
                "command": command,
                "arg": arg,
            },
        )
        assert cmd_msg.text == "/model claude-3.5-sonnet"
        assert cmd_msg.metadata["command"] == "model"
        assert cmd_msg.metadata["arg"] == "claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_all_commands_are_recognized(self):
        for cmd_name in _DISCORD_COMMANDS:
            text = f"/{cmd_name}"
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()
            assert command in _DISCORD_COMMANDS, f"/{cmd_name} not recognized"
