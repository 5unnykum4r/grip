"""Tests for Slack channel command handling."""

from __future__ import annotations

import pytest

from grip.bus.events import InboundMessage
from grip.channels.slack import _SLACK_COMMANDS


class TestSlackCommandParsing:
    def test_slack_commands_dict_has_expected_entries(self):
        expected = {"help", "new", "status", "model", "clear", "compact", "version"}
        assert set(_SLACK_COMMANDS.keys()) == expected

    def test_command_descriptions_are_strings(self):
        for _cmd, desc in _SLACK_COMMANDS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestSlackCommandHandling:
    """Test command parsing logic that on_event applies to Slack messages."""

    @pytest.mark.asyncio
    async def test_help_command_detected_with_slash(self):
        text = "/help"
        assert text.startswith(("/", "!"))
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command == "help"
        assert command in _SLACK_COMMANDS

    @pytest.mark.asyncio
    async def test_help_command_detected_with_bang(self):
        text = "!help"
        assert text.startswith(("/", "!"))
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command == "help"
        assert command in _SLACK_COMMANDS

    @pytest.mark.asyncio
    async def test_model_command_extracts_arg(self):
        text = "/model gpt-4o"
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        assert command == "model"
        assert arg == "gpt-4o"

    @pytest.mark.asyncio
    async def test_unknown_command_not_in_dict(self):
        text = "/deploy"
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        assert command not in _SLACK_COMMANDS

    @pytest.mark.asyncio
    async def test_regular_message_not_treated_as_command(self):
        text = "Hello team"
        assert not text.startswith(("/", "!"))

    @pytest.mark.asyncio
    async def test_command_creates_correct_inbound(self):
        """Simulate the InboundMessage that on_event creates for a Slack command."""
        event = {
            "text": "/new",
            "channel": "C12345",
            "user": "U67890",
            "ts": "1234567890.123",
            "team": "T99999",
        }
        text = event["text"].strip()
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        cmd_msg = InboundMessage(
            channel="slack",
            chat_id=event["channel"],
            user_id=event["user"],
            text=f"/{command} {arg}".strip(),
            metadata={
                "ts": event["ts"],
                "team": event["team"],
                "command": command,
                "arg": arg,
            },
        )
        assert cmd_msg.channel == "slack"
        assert cmd_msg.text == "/new"
        assert cmd_msg.metadata["command"] == "new"
        assert cmd_msg.metadata["arg"] == ""
        assert cmd_msg.chat_id == "C12345"
        assert cmd_msg.user_id == "U67890"

    @pytest.mark.asyncio
    async def test_command_with_arg_creates_correct_inbound(self):
        event = {
            "text": "!model claude-3.5-sonnet",
            "channel": "C12345",
            "user": "U67890",
            "ts": "123.456",
            "team": "T99999",
        }
        text = event["text"].strip()
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        cmd_msg = InboundMessage(
            channel="slack",
            chat_id=event["channel"],
            user_id=event["user"],
            text=f"/{command} {arg}".strip(),
            metadata={
                "ts": event["ts"],
                "team": event["team"],
                "command": command,
                "arg": arg,
            },
        )
        assert cmd_msg.text == "/model claude-3.5-sonnet"
        assert cmd_msg.metadata["command"] == "model"
        assert cmd_msg.metadata["arg"] == "claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_all_commands_are_recognized(self):
        for cmd_name in _SLACK_COMMANDS:
            text = f"/{cmd_name}"
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()
            assert command in _SLACK_COMMANDS, f"/{cmd_name} not recognized"

    @pytest.mark.asyncio
    async def test_slack_and_discord_commands_match(self):
        """Discord and Slack should support the same set of commands."""
        from grip.channels.discord import _DISCORD_COMMANDS

        assert set(_SLACK_COMMANDS.keys()) == set(_DISCORD_COMMANDS.keys())
