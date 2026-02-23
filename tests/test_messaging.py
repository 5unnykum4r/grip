"""Tests for messaging fixes: DirectSender, channel targeting, async detection, wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grip.channels.direct import (
    DirectSender,
    _parse_session_key,
    _unwrap_engine,
    wire_direct_sender,
)
from grip.config.schema import ChannelEntry, ChannelsConfig
from grip.tools.base import ToolContext
from grip.tools.message import MessageTool, SendFileTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channels_config(
    telegram_token: str = "",
    discord_token: str = "",
    slack_token: str = "",
) -> ChannelsConfig:
    return ChannelsConfig(
        telegram=ChannelEntry(enabled=bool(telegram_token), token=telegram_token),
        discord=ChannelEntry(enabled=bool(discord_token), token=discord_token),
        slack=ChannelEntry(enabled=bool(slack_token), token=slack_token),
    )


def _make_ctx(session_key: str = "cli:interactive") -> ToolContext:
    return ToolContext(workspace_path=Path("/tmp"), session_key=session_key)


# ---------------------------------------------------------------------------
# TestParseSessionKey
# ---------------------------------------------------------------------------


class TestParseSessionKey:
    def test_telegram_key(self):
        assert _parse_session_key("telegram:12345") == ("telegram", "12345")

    def test_discord_key(self):
        assert _parse_session_key("discord:99887766") == ("discord", "99887766")

    def test_slack_key(self):
        assert _parse_session_key("slack:C01ABCDEF") == ("slack", "C01ABCDEF")

    def test_cli_key_returns_empty(self):
        assert _parse_session_key("cli:interactive") == ("", "")

    def test_no_colon_returns_empty(self):
        assert _parse_session_key("nochannel") == ("", "")

    def test_unknown_channel_returns_empty(self):
        assert _parse_session_key("email:user@test.com") == ("", "")


# ---------------------------------------------------------------------------
# TestDirectSender
# ---------------------------------------------------------------------------


class TestDirectSender:
    def test_get_token_returns_empty_when_not_configured(self):
        config = _make_channels_config()
        sender = DirectSender(config)
        assert sender._get_token("telegram") == ""

    def test_get_token_returns_value_when_configured(self):
        config = _make_channels_config(telegram_token="bot123")
        sender = DirectSender(config)
        assert sender._get_token("telegram") == "bot123"

    def test_get_token_unknown_channel(self):
        config = _make_channels_config()
        sender = DirectSender(config)
        assert sender._get_token("email") == ""

    @pytest.mark.asyncio
    async def test_send_message_warns_on_invalid_session_key(self):
        config = _make_channels_config(telegram_token="bot123")
        sender = DirectSender(config)
        with patch("grip.channels.direct.logger") as mock_logger:
            await sender.send_message("cli:interactive", "hello")
            mock_logger.warning.assert_called()
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_message_warns_on_missing_token(self):
        config = _make_channels_config()
        sender = DirectSender(config)
        with patch("grip.channels.direct.logger") as mock_logger:
            await sender.send_message("telegram:12345", "hello")
            mock_logger.warning.assert_called()
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_message_routes_to_telegram(self):
        config = _make_channels_config(telegram_token="bot123")
        sender = DirectSender(config)
        sender._send_telegram = AsyncMock()
        await sender.send_message("telegram:12345", "hello")
        sender._send_telegram.assert_called_once_with("bot123", "12345", "hello")
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_message_routes_to_discord(self):
        config = _make_channels_config(discord_token="bot456")
        sender = DirectSender(config)
        sender._send_discord = AsyncMock()
        await sender.send_message("discord:99887766", "hello")
        sender._send_discord.assert_called_once_with("bot456", "99887766", "hello")
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_message_routes_to_slack(self):
        config = _make_channels_config(slack_token="xoxb-789")
        sender = DirectSender(config)
        sender._send_slack = AsyncMock()
        await sender.send_message("slack:C01ABC", "hello")
        sender._send_slack.assert_called_once_with("xoxb-789", "C01ABC", "hello")
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_file_warns_on_missing_file(self, tmp_path):
        config = _make_channels_config(telegram_token="bot123")
        sender = DirectSender(config)
        with patch("grip.channels.direct.logger") as mock_logger:
            await sender.send_file("telegram:12345", str(tmp_path / "nonexistent.txt"), "cap")
            mock_logger.error.assert_called()
        await sender.close()

    @pytest.mark.asyncio
    async def test_send_file_routes_to_telegram(self, tmp_path):
        config = _make_channels_config(telegram_token="bot123")
        sender = DirectSender(config)
        sender._send_telegram_file = AsyncMock()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        await sender.send_file("telegram:12345", str(test_file), "caption")
        sender._send_telegram_file.assert_called_once_with("bot123", "12345", test_file, "caption")
        await sender.close()

    @pytest.mark.asyncio
    async def test_close_shuts_down_client(self):
        config = _make_channels_config()
        sender = DirectSender(config)
        await sender.close()
        assert sender._client.is_closed


# ---------------------------------------------------------------------------
# TestMessageToolTargeting
# ---------------------------------------------------------------------------


class TestMessageToolTargeting:
    @pytest.mark.asyncio
    async def test_default_uses_ctx_session_key(self):
        callback = MagicMock()
        tool = MessageTool(callback)
        ctx = _make_ctx(session_key="telegram:12345")
        await tool.execute({"text": "hello"}, ctx)
        callback.assert_called_once_with("telegram:12345", "hello")

    @pytest.mark.asyncio
    async def test_channel_and_chat_id_override_session_key(self):
        callback = MagicMock()
        tool = MessageTool(callback)
        ctx = _make_ctx(session_key="cli:interactive")
        await tool.execute({"text": "hello", "channel": "discord", "chat_id": "999"}, ctx)
        callback.assert_called_once_with("discord:999", "hello")

    @pytest.mark.asyncio
    async def test_only_channel_without_chat_id_falls_back(self):
        callback = MagicMock()
        tool = MessageTool(callback)
        ctx = _make_ctx(session_key="cli:interactive")
        await tool.execute({"text": "hello", "channel": "discord"}, ctx)
        callback.assert_called_once_with("cli:interactive", "hello")

    @pytest.mark.asyncio
    async def test_no_callback_logs_message(self):
        tool = MessageTool()
        ctx = _make_ctx()
        result = await tool.execute({"text": "hello"}, ctx)
        assert "no active channel" in result.lower()


class TestSendFileToolTargeting:
    @pytest.mark.asyncio
    async def test_channel_override(self, tmp_path):
        callback = MagicMock()
        tool = SendFileTool(callback)
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")
        ctx = _make_ctx(session_key="cli:interactive")
        await tool.execute(
            {"file_path": str(test_file), "channel": "telegram", "chat_id": "123"}, ctx
        )
        callback.assert_called_once_with("telegram:123", str(test_file), "")

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        callback = MagicMock()
        tool = SendFileTool(callback)
        ctx = _make_ctx()
        result = await tool.execute({"file_path": "/nonexistent/file.txt"}, ctx)
        assert "not found" in result.lower()
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# TestAsyncDetection
# ---------------------------------------------------------------------------


class TestAsyncDetection:
    @pytest.mark.asyncio
    async def test_sync_callback_called_directly(self):
        callback = MagicMock()
        tool = MessageTool(callback)
        ctx = _make_ctx(session_key="telegram:123")
        await tool.execute({"text": "test"}, ctx)
        callback.assert_called_once_with("telegram:123", "test")

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self):
        callback = AsyncMock()
        tool = MessageTool(callback)
        ctx = _make_ctx(session_key="telegram:123")
        await tool.execute({"text": "test"}, ctx)
        callback.assert_awaited_once_with("telegram:123", "test")

    @pytest.mark.asyncio
    async def test_send_file_sync_callback(self, tmp_path):
        callback = MagicMock()
        tool = SendFileTool(callback)
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")
        ctx = _make_ctx(session_key="telegram:123")
        await tool.execute({"file_path": str(test_file)}, ctx)
        callback.assert_called_once_with("telegram:123", str(test_file), "")

    @pytest.mark.asyncio
    async def test_send_file_async_callback(self, tmp_path):
        callback = AsyncMock()
        tool = SendFileTool(callback)
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")
        ctx = _make_ctx(session_key="telegram:123")
        await tool.execute({"file_path": str(test_file)}, ctx)
        callback.assert_awaited_once_with("telegram:123", str(test_file), "")


# ---------------------------------------------------------------------------
# TestUnwrapEngine
# ---------------------------------------------------------------------------


class TestUnwrapEngine:
    def test_returns_same_if_no_inner(self):
        engine = MagicMock(spec=[])
        assert _unwrap_engine(engine) is engine

    def test_unwraps_single_layer(self):
        inner = MagicMock(spec=[])
        outer = MagicMock()
        outer._inner = inner
        del inner._inner
        assert _unwrap_engine(outer) is inner

    def test_unwraps_double_layer(self):
        core = MagicMock(spec=[])
        middle = MagicMock()
        middle._inner = core
        outer = MagicMock()
        outer._inner = middle
        del core._inner
        assert _unwrap_engine(outer) is core


# ---------------------------------------------------------------------------
# TestWireDirectSender
# ---------------------------------------------------------------------------


class TestWireDirectSender:
    def test_returns_none_when_no_tokens(self):
        config = _make_channels_config()
        engine = MagicMock()
        result = wire_direct_sender(engine, config)
        assert result is None

    def test_wires_sdk_runner(self):
        from grip.engines.sdk_engine import SDKRunner

        config = _make_channels_config(telegram_token="bot123")
        runner = MagicMock(spec=SDKRunner)
        del runner._inner
        sender = wire_direct_sender(runner, config)
        assert sender is not None
        runner.set_send_callback.assert_called_once()
        runner.set_send_file_callback.assert_called_once()

    def test_wires_litellm_runner(self):
        from grip.engines.litellm_engine import LiteLLMRunner

        config = _make_channels_config(discord_token="bot456")
        msg_tool = MessageTool()
        file_tool = SendFileTool()
        registry = MagicMock()
        registry.get.side_effect = lambda name: {"send_message": msg_tool, "send_file": file_tool}.get(name)
        runner = MagicMock(spec=LiteLLMRunner)
        runner.registry = registry
        del runner._inner
        sender = wire_direct_sender(runner, config)
        assert sender is not None
        assert msg_tool._callback is not None
        assert file_tool._callback is not None

    def test_unwraps_engine_chain(self):
        from grip.engines.sdk_engine import SDKRunner

        config = _make_channels_config(slack_token="xoxb-test")
        inner_runner = MagicMock(spec=SDKRunner)
        del inner_runner._inner
        wrapper = MagicMock()
        wrapper._inner = inner_runner
        sender = wire_direct_sender(wrapper, config)
        assert sender is not None
        inner_runner.set_send_callback.assert_called_once()
        inner_runner.set_send_file_callback.assert_called_once()


# ---------------------------------------------------------------------------
# TestMessageToolParameters
# ---------------------------------------------------------------------------


class TestMessageToolParameters:
    def test_message_tool_has_channel_param(self):
        tool = MessageTool()
        props = tool.parameters["properties"]
        assert "channel" in props
        assert "chat_id" in props

    def test_send_file_tool_has_channel_param(self):
        tool = SendFileTool()
        props = tool.parameters["properties"]
        assert "channel" in props
        assert "chat_id" in props

    def test_text_still_required(self):
        tool = MessageTool()
        assert "text" in tool.parameters["required"]

    def test_channel_not_required(self):
        tool = MessageTool()
        assert "channel" not in tool.parameters.get("required", [])
