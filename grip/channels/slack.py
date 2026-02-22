"""Slack channel integration using slack-sdk Socket Mode.

Socket Mode eliminates the need for a public URL or ngrok — the bot
connects outbound to Slack's servers via WebSocket.

Requires: pip install grip[channels-slack]
Config:
  config.channels.slack.enabled = true
  config.channels.slack.token = "xoxb-BOT-TOKEN"
  config.channels.slack.extra.app_token = "xapp-APP-LEVEL-TOKEN"
"""

from __future__ import annotations

import asyncio
import contextlib

from loguru import logger

from grip.bus.events import InboundMessage
from grip.bus.queue import MessageBus
from grip.channels.base import BaseChannel
from grip.config.schema import ChannelEntry

SLACK_MAX_MESSAGE_LENGTH = 40000

_SLACK_COMMANDS = {
    "help": "List available commands",
    "new": "Start a fresh conversation",
    "status": "Show session info",
    "model": "Show or switch AI model",
    "clear": "Clear conversation history",
    "compact": "Summarize and compress history",
    "version": "Show grip version",
}


class SlackChannel(BaseChannel):
    """Slack bot channel using Socket Mode (no public URL needed)."""

    def __init__(self, config: ChannelEntry) -> None:
        super().__init__(config)
        self._socket_client = None
        self._web_client = None
        self._task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "slack"

    async def start(self, bus: MessageBus) -> None:
        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as exc:
            raise RuntimeError(
                "slack-sdk is required for Slack channel. "
                "Install with: uv pip install grip[channels-slack]"
            ) from exc

        self._bus = bus
        bus.subscribe_outbound(self._handle_outbound)

        bot_token = self._config.token
        app_token = self._config.extra.get("app_token", "")

        if not bot_token:
            raise ValueError("Slack bot token is required (config.channels.slack.token)")
        if not app_token:
            raise ValueError(
                "Slack app-level token is required for Socket Mode "
                "(config.channels.slack.extra.app_token)"
            )

        self._web_client = AsyncWebClient(token=bot_token)
        self._socket_client = SocketModeClient(
            app_token=app_token,
            web_client=self._web_client,
        )
        channel_ref = self

        async def on_event(client, req):
            """Handle incoming Socket Mode events."""
            if req.type == "events_api":
                event = req.payload.get("event", {})
                if event.get("type") == "message" and "subtype" not in event:
                    user_id = event.get("user", "")
                    if not channel_ref.is_allowed(user_id):
                        logger.warning("Slack: blocked message from non-allowed user {}", user_id)
                        await client.send_socket_mode_response({"envelope_id": req.envelope_id})
                        return

                    text = event.get("text", "").strip()
                    chat_id = event.get("channel", "")

                    if text.startswith(("!", "/")):
                        parts = text[1:].split(maxsplit=1)
                        command = parts[0].lower()
                        arg = parts[1] if len(parts) > 1 else ""
                        if command in _SLACK_COMMANDS:
                            cmd_msg = InboundMessage(
                                channel="slack",
                                chat_id=chat_id,
                                user_id=user_id,
                                text=f"/{command} {arg}".strip(),
                                metadata={
                                    "ts": event.get("ts", ""),
                                    "team": event.get("team", ""),
                                    "command": command,
                                    "arg": arg,
                                },
                            )
                            await bus.push_inbound(cmd_msg)
                            await client.send_socket_mode_response({"envelope_id": req.envelope_id})
                            return

                    msg = InboundMessage(
                        channel="slack",
                        chat_id=chat_id,
                        user_id=user_id,
                        text=text,
                        metadata={
                            "ts": event.get("ts", ""),
                            "team": event.get("team", ""),
                        },
                    )
                    await bus.push_inbound(msg)

                await client.send_socket_mode_response({"envelope_id": req.envelope_id})

        self._socket_client.socket_mode_request_listeners.append(on_event)
        self._task = asyncio.create_task(self._socket_client.connect(), name="slack-socket-mode")
        logger.info("Slack channel started (Socket Mode)")

    async def stop(self) -> None:
        if self._socket_client:
            await self._socket_client.close()
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Slack channel stopped")

    async def send(self, chat_id: str, text: str, **kwargs) -> None:
        if not self._web_client:
            logger.error("Slack: cannot send, web client not initialized")
            return

        chunks = self.split_message(text, SLACK_MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            await self._web_client.chat_postMessage(channel=chat_id, text=chunk)
