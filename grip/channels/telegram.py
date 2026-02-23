"""Telegram channel integration using python-telegram-bot (async native).

Requires: pip install grip[channels-telegram]
Config: config.channels.telegram.enabled = true, .token = "BOT_TOKEN"
Optional: config.channels.telegram.allow_from = ["123456789"]

Bot commands registered with Telegram:
  /start   — Welcome message
  /help    — List available commands
  /new     — Start a fresh conversation
  /status  — Show session info
  /model   — Show or switch AI model (/model gpt-4o)
  /undo    — Remove last exchange
  /clear   — Clear conversation history
  /compact — Summarize and compress session history

All text messages (non-command) are forwarded to the agent loop.
Photo captions and document captions are also processed.
"""

from __future__ import annotations

import contextlib
import html
import re

from loguru import logger

from grip.bus.events import InboundMessage
from grip.bus.queue import MessageBus
from grip.channels.base import BaseChannel
from grip.config.schema import ChannelEntry

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def _parse_chat_id(chat_id: str) -> int | str:
    """Convert chat_id to int if numeric, otherwise return as-is for @channel names."""
    try:
        return int(chat_id)
    except ValueError:
        return chat_id

# Telegram HTML mode only supports a small set of tags. We convert
# common Markdown patterns the LLM produces into HTML equivalents.
_MD_TO_HTML_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Code blocks: ```lang\ncode\n``` -> <pre><code>code</code></pre>
    (re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL), r"<pre><code>\1</code></pre>"),
    # Inline code: `code` -> <code>code</code>
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    # Bold: **text** or __text__ -> <b>text</b>
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"__(.+?)__"), r"<b>\1</b>"),
    # Italic: *text* or _text_ -> <i>text</i>
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"), r"<i>\1</i>"),
    # Strikethrough: ~~text~~ -> <s>text</s>
    (re.compile(r"~~(.+?)~~"), r"<s>\1</s>"),
    # Links: [text](url) -> <a href="url">text</a> (only http/https to prevent javascript: injection)
    (re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)"), r'<a href="\2">\1</a>'),
]

# Single source of truth for bot commands.
# Used by: BotFather registration, /help text generation, and CommandHandler setup.
_BOT_COMMANDS = [
    ("start", "Welcome message"),
    ("help", "List available commands"),
    ("new", "Start a fresh conversation"),
    ("status", "Show session info"),
    ("model", "Show or switch AI model"),
    ("trust", "Trust a directory (e.g. /trust ~/Downloads)"),
    ("undo", "Remove last exchange"),
    ("clear", "Clear conversation history"),
    ("compact", "Summarize and compress history"),
]


def _escape_html(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(text, quote=False)


def _markdown_to_telegram_html(text: str) -> str:
    """Convert LLM Markdown output to Telegram-safe HTML.

    First escapes all HTML entities, then converts known Markdown patterns
    to the subset of HTML tags Telegram supports: <b>, <i>, <s>, <code>,
    <pre>, <a href>.
    """
    escaped = _escape_html(text)

    for pattern, replacement in _MD_TO_HTML_PATTERNS:
        escaped = pattern.sub(replacement, escaped)

    return escaped


def _build_help_text() -> str:
    """Generate /help response from _BOT_COMMANDS (single source of truth)."""
    lines = ["<b>Available Commands</b>\n"]
    for cmd, desc in _BOT_COMMANDS:
        lines.append(f"/{cmd} — {_escape_html(desc)}")
    lines.append("\nSend any text message to chat with the AI.")
    return "\n".join(lines)


class TelegramChannel(BaseChannel):
    """Telegram bot channel via python-telegram-bot library."""

    def __init__(self, config: ChannelEntry) -> None:
        super().__init__(config)
        self._app = None

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self, bus: MessageBus) -> None:
        try:
            from telegram import BotCommand, Update
            from telegram.constants import ChatAction
            from telegram.ext import (
                ApplicationBuilder,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError as exc:
            raise RuntimeError(
                "python-telegram-bot is required for Telegram channel. "
                "Install with: uv pip install grip[channels-telegram]"
            ) from exc

        self._bus = bus
        bus.subscribe_outbound(self._handle_outbound)

        token = self._config.token.get_secret_value()
        if not token:
            raise ValueError("Telegram bot token is required (config.channels.telegram.token)")

        self._app = ApplicationBuilder().token(token).build()
        channel_ref = self

        # ── Helper: check user permission and extract IDs ──
        def _check_user(update: Update) -> tuple[str, str] | None:
            """Return (chat_id, user_id) if allowed, or None if blocked."""
            user_id = str(update.effective_user.id) if update.effective_user else ""
            if not channel_ref.is_allowed(user_id):
                logger.warning("Telegram: blocked from non-allowed user {}", user_id)
                return None
            chat_id = str(update.effective_chat.id) if update.effective_chat else ""
            return chat_id, user_id

        # ── Helper: push a control command to the bus ──
        async def _push_command(
            chat_id: str, user_id: str, command: str, **extra_meta: str
        ) -> None:
            meta: dict = {"command": command}
            meta.update(extra_meta)
            msg = InboundMessage(
                channel="telegram",
                chat_id=chat_id,
                user_id=user_id,
                text=f"/{command}",
                metadata=meta,
            )
            await bus.push_inbound(msg)

        # ── /start ──
        async def cmd_start(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            user = update.effective_user
            name = user.first_name if user else "there"
            await update.effective_chat.send_message(
                f"<b>Hey {_escape_html(name)}!</b>\n\n"
                "I'm <b>grip</b> — your AI assistant.\n\n"
                "Send me any message and I'll do my best to help.\n"
                "Type /help to see all available commands.",
                parse_mode="HTML",
            )

        # ── /help ──
        async def cmd_help(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await update.effective_chat.send_message(
                _build_help_text(),
                parse_mode="HTML",
            )

        # ── /new — route through bus ──
        async def cmd_new(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await _push_command(ids[0], ids[1], "new")
            await update.effective_chat.send_message(
                "Session cleared. Starting fresh conversation.",
                parse_mode="HTML",
            )

        # ── /status — route through bus (gateway has session_mgr access) ──
        async def cmd_status(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await _push_command(ids[0], ids[1], "status")

        # ── /model [name] — route through bus (gateway stores + applies) ──
        async def cmd_model(update: Update, _ctx) -> None:
            if not update.effective_chat or not update.message:
                return
            ids = _check_user(update)
            if ids is None:
                return
            text = update.message.text or ""
            parts = text.strip().split(maxsplit=1)
            model_name = parts[1].strip() if len(parts) > 1 else ""
            await _push_command(ids[0], ids[1], "model", model_name=model_name)

        # ── /undo — route through bus ──
        async def cmd_undo(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await _push_command(ids[0], ids[1], "undo")

        # ── /clear — route through bus ──
        async def cmd_clear(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await _push_command(ids[0], ids[1], "clear")
            await update.effective_chat.send_message(
                "Conversation history cleared.",
                parse_mode="HTML",
            )

        # ── /compact — route through bus ──
        async def cmd_compact(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            await _push_command(ids[0], ids[1], "compact")
            await update.effective_chat.send_message(
                "Compacting session history...",
                parse_mode="HTML",
            )

        # ── /trust — route through bus ──
        async def cmd_trust(update: Update, _ctx) -> None:
            if not update.effective_chat:
                return
            ids = _check_user(update)
            if ids is None:
                return
            text = (update.message.text or "").strip()
            trust_path = text.split(maxsplit=1)[1] if " " in text else ""
            await _push_command(ids[0], ids[1], "trust", trust_path=trust_path)

        # ── Text messages ──
        async def on_message(update: Update, _ctx) -> None:
            if not update.message or not update.message.text:
                return
            ids = _check_user(update)
            if ids is None:
                return

            # Show "typing..." indicator while the agent processes
            if update.effective_chat:
                with contextlib.suppress(Exception):
                    await update.effective_chat.send_action(ChatAction.TYPING)

            msg = InboundMessage(
                channel="telegram",
                chat_id=ids[0],
                user_id=ids[1],
                text=update.message.text,
                metadata={"message_id": str(update.message.message_id)},
            )
            await bus.push_inbound(msg)

        # ── Photo messages (process caption) ──
        async def on_photo(update: Update, _ctx) -> None:
            if not update.message:
                return
            ids = _check_user(update)
            if ids is None:
                return
            caption = update.message.caption or "[User sent a photo without caption]"

            msg = InboundMessage(
                channel="telegram",
                chat_id=ids[0],
                user_id=ids[1],
                text=caption,
                metadata={
                    "message_id": str(update.message.message_id),
                    "type": "photo",
                },
            )
            await bus.push_inbound(msg)

        # ── Document messages (process caption) ──
        async def on_document(update: Update, _ctx) -> None:
            if not update.message:
                return
            ids = _check_user(update)
            if ids is None:
                return
            doc = update.message.document
            doc_name = doc.file_name if doc else "unknown"
            caption = update.message.caption or ""
            text = f"[User sent document: {doc_name}]"
            if caption:
                text += f"\n{caption}"

            msg = InboundMessage(
                channel="telegram",
                chat_id=ids[0],
                user_id=ids[1],
                text=text,
                metadata={
                    "message_id": str(update.message.message_id),
                    "type": "document",
                    "file_name": doc_name,
                },
            )
            await bus.push_inbound(msg)

        # ── Voice messages ──
        async def on_voice(update: Update, _ctx) -> None:
            if not update.message:
                return
            ids = _check_user(update)
            if ids is None:
                return
            duration = update.message.voice.duration if update.message.voice else 0

            msg = InboundMessage(
                channel="telegram",
                chat_id=ids[0],
                user_id=ids[1],
                text=f"[User sent a voice message ({duration}s). Voice transcription is not yet supported.]",
                metadata={
                    "message_id": str(update.message.message_id),
                    "type": "voice",
                    "duration": duration,
                },
            )
            await bus.push_inbound(msg)

        # ── Unknown command handler ──
        async def on_unknown_command(update: Update, _ctx) -> None:
            if not update.effective_chat or not update.message:
                return
            ids = _check_user(update)
            if ids is None:
                return
            cmd = (update.message.text or "").split()[0]
            await update.effective_chat.send_message(
                f"Unknown command: <code>{_escape_html(cmd)}</code>\n"
                "Type /help for available commands.",
                parse_mode="HTML",
            )

        # Register handlers (order matters — commands first, then messages)
        command_handlers = {
            "start": cmd_start,
            "help": cmd_help,
            "new": cmd_new,
            "status": cmd_status,
            "model": cmd_model,
            "trust": cmd_trust,
            "undo": cmd_undo,
            "clear": cmd_clear,
            "compact": cmd_compact,
        }
        for cmd_name, handler in command_handlers.items():
            self._app.add_handler(CommandHandler(cmd_name, handler))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
        self._app.add_handler(MessageHandler(filters.PHOTO, on_photo))
        self._app.add_handler(MessageHandler(filters.Document.ALL, on_document))
        self._app.add_handler(MessageHandler(filters.VOICE, on_voice))
        self._app.add_handler(MessageHandler(filters.COMMAND, on_unknown_command))

        await self._app.initialize()
        await self._app.start()

        # Register bot commands with Telegram so they appear in the menu
        try:
            await self._app.bot.set_my_commands(
                [BotCommand(cmd, desc) for cmd, desc in _BOT_COMMANDS]
            )
        except Exception as exc:
            logger.warning("Failed to register Telegram bot commands: {}", exc)

        if self._app.updater:
            await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram channel started with {} command handlers", len(command_handlers))

    async def stop(self) -> None:
        if self._app:
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram channel stopped")

    async def send(self, chat_id: str, text: str, **kwargs) -> None:
        if not self._app or not self._app.bot:
            logger.error("Telegram: cannot send, bot not initialized")
            return

        html_text = _markdown_to_telegram_html(text)
        chunks = self.split_message(html_text, TELEGRAM_MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                await self._app.bot.send_message(
                    chat_id=_parse_chat_id(chat_id),
                    text=chunk,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as exc:
                # Fallback: if HTML parsing fails, send as plain text
                logger.warning("Telegram HTML send failed, falling back to plain text: {}", exc)
                plain_chunk = re.sub(r"<[^>]+>", "", chunk)
                try:
                    await self._app.bot.send_message(
                        chat_id=_parse_chat_id(chat_id),
                        text=plain_chunk,
                    )
                except Exception as fallback_exc:
                    logger.error("Telegram send failed completely: {}", fallback_exc)

    async def send_file(self, chat_id: str, file_path: str, caption: str = "") -> None:
        """Send a file to Telegram as a photo (images) or document (everything else).

        Supports: PNG, JPG, JPEG, GIF, WEBP as photos. All other files sent as documents.
        Captions are converted to Telegram HTML and truncated to 1024 chars (Telegram limit).
        """
        from pathlib import Path

        if not self._app or not self._app.bot:
            logger.error("Telegram: cannot send file, bot not initialized")
            return

        path = Path(file_path)
        if not path.is_file():
            logger.error("Telegram: file not found: {}", file_path)
            await self.send(chat_id, f"File not found: {file_path}")
            return

        html_caption = _markdown_to_telegram_html(caption)[:1024] if caption else ""
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        is_image = path.suffix.lower() in image_extensions

        try:
            with open(path, "rb") as f:
                if is_image:
                    await self._app.bot.send_photo(
                        chat_id=_parse_chat_id(chat_id),
                        photo=f,
                        caption=html_caption or None,
                        parse_mode="HTML" if html_caption else None,
                    )
                else:
                    await self._app.bot.send_document(
                        chat_id=_parse_chat_id(chat_id),
                        document=f,
                        filename=path.name,
                        caption=html_caption or None,
                        parse_mode="HTML" if html_caption else None,
                    )
            logger.info(
                "Telegram: sent {} to chat {}", "photo" if is_image else "document", chat_id
            )
        except Exception as exc:
            logger.error("Telegram: failed to send file {}: {}", file_path, exc)
            # Fallback: try without caption parsing
            try:
                with open(path, "rb") as f:
                    if is_image:
                        await self._app.bot.send_photo(
                            chat_id=_parse_chat_id(chat_id),
                            photo=f,
                            caption=caption[:1024] if caption else None,
                        )
                    else:
                        await self._app.bot.send_document(
                            chat_id=_parse_chat_id(chat_id),
                            document=f,
                            filename=path.name,
                            caption=caption[:1024] if caption else None,
                        )
            except Exception as fallback_exc:
                logger.error("Telegram: file send failed completely: {}", fallback_exc)
                await self.send(chat_id, f"Failed to send file: {path.name}")
