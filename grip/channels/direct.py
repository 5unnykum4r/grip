"""Lightweight direct sender for CLI/API modes.

Uses httpx to call Telegram/Discord/Slack HTTP APIs directly,
without starting heavy channel infrastructure (polling, websockets,
Socket Mode). This enables agents running in CLI or API mode to
send messages to chat platforms when channel tokens are configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from grip.channels.base import BaseChannel
from grip.config.schema import ChannelsConfig

TELEGRAM_API = "https://api.telegram.org/bot{token}"
DISCORD_API = "https://discord.com/api/v10"
SLACK_API = "https://slack.com/api"

TELEGRAM_MAX_LENGTH = 4096
DISCORD_MAX_LENGTH = 2000
SLACK_MAX_LENGTH = 4000


def _parse_session_key(session_key: str) -> tuple[str, str]:
    """Split 'channel:chat_id' into (channel, chat_id).

    Returns ('', '') if the format is invalid (e.g. 'cli:interactive').
    """
    parts = session_key.split(":", 1)
    if len(parts) == 2 and parts[0] in ("telegram", "discord", "slack"):
        return parts[0], parts[1]
    return "", ""


class DirectSender:
    """Sends messages to Telegram/Discord/Slack via HTTP APIs.

    Designed for CLI and API modes where the full channel infrastructure
    (MessageBus, polling, websockets) is not running. Only requires
    channel tokens from config — no optional deps needed.
    """

    def __init__(self, channels_config: ChannelsConfig) -> None:
        self._config = channels_config
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Shut down the HTTP client."""
        await self._client.aclose()

    def _get_token(self, channel: str) -> str:
        """Get the bot token for a channel, or '' if not configured."""
        entry = getattr(self._config, channel, None)
        if entry and entry.token:
            value = entry.token.get_secret_value()
            return value if value else ""
        return ""

    async def send_message(self, session_key: str, text: str) -> None:
        """Route a message to the correct channel API."""
        channel, chat_id = _parse_session_key(session_key)
        if not channel:
            logger.warning("DirectSender: cannot route session_key '{}'", session_key)
            return

        token = self._get_token(channel)
        if not token:
            logger.warning("DirectSender: no token configured for '{}'", channel)
            return

        if channel == "telegram":
            await self._send_telegram(token, chat_id, text)
        elif channel == "discord":
            await self._send_discord(token, chat_id, text)
        elif channel == "slack":
            await self._send_slack(token, chat_id, text)

    async def send_file(self, session_key: str, file_path: str, caption: str) -> None:
        """Route a file send to the correct channel API."""
        channel, chat_id = _parse_session_key(session_key)
        if not channel:
            logger.warning("DirectSender: cannot route session_key '{}' for file", session_key)
            return

        token = self._get_token(channel)
        if not token:
            logger.warning("DirectSender: no token configured for '{}' (file)", channel)
            return

        path = Path(file_path)
        if not path.is_file():
            logger.error("DirectSender: file not found: {}", file_path)
            return

        if channel == "telegram":
            await self._send_telegram_file(token, chat_id, path, caption)
        elif channel == "discord":
            await self._send_discord_file(token, chat_id, path, caption)
        elif channel == "slack":
            await self._send_slack_file(token, chat_id, path, caption)

    # -- Telegram ----------------------------------------------------------

    async def _send_telegram(self, token: str, chat_id: str, text: str) -> None:
        url = f"{TELEGRAM_API.format(token=token)}/sendMessage"
        chunks = BaseChannel.split_message(text, TELEGRAM_MAX_LENGTH)
        for chunk in chunks:
            try:
                resp = await self._client.post(url, json={"chat_id": chat_id, "text": chunk})
                if resp.status_code != 200:
                    logger.error("Telegram sendMessage failed ({}): {}", resp.status_code, resp.text)
            except httpx.HTTPError as exc:
                logger.error("Telegram sendMessage error: {}", exc)

    async def _send_telegram_file(
        self, token: str, chat_id: str, path: Path, caption: str
    ) -> None:
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        is_image = path.suffix.lower() in image_exts
        endpoint = "sendPhoto" if is_image else "sendDocument"
        field_name = "photo" if is_image else "document"
        url = f"{TELEGRAM_API.format(token=token)}/{endpoint}"

        try:
            with open(path, "rb") as f:
                files: dict[str, Any] = {field_name: (path.name, f, "application/octet-stream")}
                data: dict[str, str] = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                resp = await self._client.post(url, data=data, files=files)
                if resp.status_code != 200:
                    logger.error("Telegram {} failed ({}): {}", endpoint, resp.status_code, resp.text)
        except (httpx.HTTPError, OSError) as exc:
            logger.error("Telegram {} error: {}", endpoint, exc)

    # -- Discord -----------------------------------------------------------

    async def _send_discord(self, token: str, chat_id: str, text: str) -> None:
        url = f"{DISCORD_API}/channels/{chat_id}/messages"
        headers = {"Authorization": f"Bot {token}"}
        chunks = BaseChannel.split_message(text, DISCORD_MAX_LENGTH)
        for chunk in chunks:
            try:
                resp = await self._client.post(url, headers=headers, json={"content": chunk})
                if resp.status_code not in (200, 201):
                    logger.error("Discord send failed ({}): {}", resp.status_code, resp.text)
            except httpx.HTTPError as exc:
                logger.error("Discord send error: {}", exc)

    async def _send_discord_file(
        self, token: str, chat_id: str, path: Path, caption: str
    ) -> None:
        url = f"{DISCORD_API}/channels/{chat_id}/messages"
        headers = {"Authorization": f"Bot {token}"}
        try:
            with open(path, "rb") as f:
                files = {"files[0]": (path.name, f, "application/octet-stream")}
                data: dict[str, str] = {}
                if caption:
                    data["payload_json"] = f'{{"content": {caption!r}}}'
                resp = await self._client.post(url, headers=headers, data=data, files=files)
                if resp.status_code not in (200, 201):
                    logger.error("Discord file send failed ({}): {}", resp.status_code, resp.text)
        except (httpx.HTTPError, OSError) as exc:
            logger.error("Discord file send error: {}", exc)

    # -- Slack -------------------------------------------------------------

    async def _send_slack(self, token: str, chat_id: str, text: str) -> None:
        url = f"{SLACK_API}/chat.postMessage"
        headers = {"Authorization": f"Bearer {token}"}
        chunks = BaseChannel.split_message(text, SLACK_MAX_LENGTH)
        for chunk in chunks:
            try:
                resp = await self._client.post(
                    url, headers=headers, json={"channel": chat_id, "text": chunk}
                )
                if resp.status_code != 200:
                    logger.error("Slack send failed ({}): {}", resp.status_code, resp.text)
                else:
                    body = resp.json()
                    if not body.get("ok"):
                        logger.error("Slack API error: {}", body.get("error", "unknown"))
            except httpx.HTTPError as exc:
                logger.error("Slack send error: {}", exc)

    async def _send_slack_file(
        self, token: str, chat_id: str, path: Path, caption: str
    ) -> None:
        url = f"{SLACK_API}/files.uploadV2"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with open(path, "rb") as f:
                resp = await self._client.post(
                    url,
                    headers=headers,
                    data={
                        "channel_id": chat_id,
                        "filename": path.name,
                        "initial_comment": caption or "",
                    },
                    files={"file": (path.name, f, "application/octet-stream")},
                )
                if resp.status_code != 200:
                    logger.error("Slack file upload failed ({}): {}", resp.status_code, resp.text)
                else:
                    body = resp.json()
                    if not body.get("ok"):
                        logger.error("Slack file upload error: {}", body.get("error", "unknown"))
        except (httpx.HTTPError, OSError) as exc:
            logger.error("Slack file upload error: {}", exc)


def _unwrap_engine(engine: Any) -> Any:
    """Walk the engine wrapper chain (TrackedEngine → LearningEngine → actual).

    Returns the innermost engine.
    """
    current = engine
    for _ in range(10):
        inner = getattr(current, "_inner", None)
        if inner is None:
            return current
        current = inner
    return current


def wire_direct_sender(engine: Any, channels_config: ChannelsConfig) -> DirectSender | None:
    """Wire a DirectSender into the engine for CLI/API modes.

    Unwraps the engine wrapper chain and sets send callbacks on
    SDKRunner or LiteLLMRunner. Returns the DirectSender instance
    (caller should close it on shutdown) or None if no channels are enabled.
    """
    has_any_token = any(
        getattr(channels_config, ch).token.get_secret_value()
        for ch in ("telegram", "discord", "slack")
    )
    if not has_any_token:
        return None

    sender = DirectSender(channels_config)
    inner = _unwrap_engine(engine)

    from grip.engines.litellm_engine import LiteLLMRunner
    from grip.engines.sdk_engine import SDKRunner
    from grip.tools.message import MessageTool, SendFileTool

    if isinstance(inner, SDKRunner):
        inner.set_send_callback(sender.send_message)
        inner.set_send_file_callback(sender.send_file)
        logger.info("DirectSender wired to SDKRunner for CLI/API messaging")
    elif isinstance(inner, LiteLLMRunner):
        registry = getattr(inner, "registry", None) or getattr(inner, "_registry", None)
        if registry:
            msg_tool = registry.get("send_message")
            if isinstance(msg_tool, MessageTool):
                msg_tool.set_callback(sender.send_message)
            file_tool = registry.get("send_file")
            if isinstance(file_tool, SendFileTool):
                file_tool.set_callback(sender.send_file)
            logger.info("DirectSender wired to LiteLLMRunner for CLI/API messaging")

    return sender
