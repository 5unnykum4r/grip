"""Message and file tools: lets the agent send messages and files back to the user.

Used by subagents, long-running tasks, and channel-connected sessions to
communicate results without waiting for the full agent loop to finish.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext

# Callback type: async function that receives (session_key, text)
MessageCallback = Callable[[str, str], Any]

# File callback type: async function that receives (session_key, file_path, caption)
FileCallback = Callable[[str, str, str], Any]


class MessageTool(Tool):
    """Send a message to the user through the active channel.

    The actual send is delegated to a callback registered at construction.
    If no callback is set (e.g. in CLI mode), the message is logged
    and returned as confirmation.
    """

    def __init__(self, callback: MessageCallback | None = None) -> None:
        self._callback = callback

    def set_callback(self, callback: MessageCallback) -> None:
        self._callback = callback

    @property
    def category(self) -> str:
        return "messaging"

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return "Send a text message to the user immediately via the configured channel."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Message text to send to the user.",
                },
                "channel": {
                    "type": "string",
                    "description": (
                        "Target channel (telegram, discord, slack). "
                        "Omit to use the current session's channel."
                    ),
                },
                "chat_id": {
                    "type": "string",
                    "description": (
                        "Target chat/channel ID on the platform. "
                        "Required when channel is specified."
                    ),
                },
            },
            "required": ["text"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        text = params["text"]
        channel = params.get("channel")
        chat_id = params.get("chat_id")
        target_key = f"{channel}:{chat_id}" if channel and chat_id else ctx.session_key

        if self._callback:
            try:
                if inspect.iscoroutinefunction(self._callback):
                    await self._callback(target_key, text)
                else:
                    self._callback(target_key, text)
                logger.debug("Message sent via callback: {}...", text[:80])
                return f"Message sent: {text[:100]}"
            except Exception as exc:
                logger.error("Message callback failed: {}", exc)
                return f"Error sending message: {exc}"

        logger.info("[Agent Message] {}", text)
        return f"Message logged (no active channel): {text[:100]}"


class SendFileTool(Tool):
    """Send a file to the user through the active channel.

    Sends images as photos and other files as document attachments.
    When running in a channel (Telegram, Discord, Slack), the file is
    delivered directly to the chat. In CLI mode, the file path is logged.

    Use this instead of telling the user to check a local folder.
    """

    def __init__(self, callback: FileCallback | None = None) -> None:
        self._callback = callback

    def set_callback(self, callback: FileCallback) -> None:
        self._callback = callback

    @property
    def category(self) -> str:
        return "messaging"

    @property
    def name(self) -> str:
        return "send_file"

    @property
    def description(self) -> str:
        return "Send a file to the user as an attachment. Provide the absolute path to an existing file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to send.",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption or description for the file.",
                },
                "channel": {
                    "type": "string",
                    "description": (
                        "Target channel (telegram, discord, slack). "
                        "Omit to use the current session's channel."
                    ),
                },
                "chat_id": {
                    "type": "string",
                    "description": (
                        "Target chat/channel ID on the platform. "
                        "Required when channel is specified."
                    ),
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        from pathlib import Path

        file_path = params["file_path"]
        caption = params.get("caption", "")
        channel = params.get("channel")
        chat_id = params.get("chat_id")
        target_key = f"{channel}:{chat_id}" if channel and chat_id else ctx.session_key

        path = Path(file_path)
        if not path.is_file():
            return f"Error: File not found: {file_path}"

        if self._callback:
            try:
                if inspect.iscoroutinefunction(self._callback):
                    await self._callback(target_key, file_path, caption)
                else:
                    self._callback(target_key, file_path, caption)
                logger.debug("File sent via callback: {}", path.name)
                return f"File sent: {path.name}"
            except Exception as exc:
                logger.error("File send callback failed: {}", exc)
                return f"Error sending file: {exc}"

        logger.info("[Agent File] {}", file_path)
        return f"File ready (no active channel): {file_path}"


def create_message_tools(
    callback: MessageCallback | None = None,
    file_callback: FileCallback | None = None,
) -> list[Tool]:
    return [MessageTool(callback), SendFileTool(file_callback)]
