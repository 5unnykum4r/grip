"""Heartbeat service: periodic autonomous agent wake-up.

Reads HEARTBEAT.md from the workspace at a configurable interval and
sends its contents to the engine as a user message. This allows
the agent to perform periodic self-directed tasks like checking
system health, summarizing recent activity, or running maintenance.

If HEARTBEAT.md is missing or empty, the heartbeat is silently skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from loguru import logger

from grip.config.schema import HeartbeatConfig
from grip.engines.types import EngineProtocol

SESSION_KEY = "heartbeat:periodic"


class HeartbeatService:
    """Periodically reads HEARTBEAT.md and feeds it to the engine."""

    def __init__(
        self,
        workspace_root: Path,
        engine: EngineProtocol,
        config: HeartbeatConfig,
        bus: Any | None = None,
        reply_to: str = "",
    ) -> None:
        self._workspace_root = workspace_root
        self._heartbeat_file = workspace_root / "HEARTBEAT.md"
        self._engine = engine
        self._config = config
        self._bus = bus
        self._reply_to = reply_to
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the heartbeat loop. Runs until cancelled."""
        if not self._config.enabled:
            logger.debug("Heartbeat service disabled")
            return

        self._stop_event.clear()
        interval = self._config.interval_minutes * 60
        logger.info("Heartbeat service started (interval: {}min)", self._config.interval_minutes)

        while not self._stop_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            if not self._stop_event.is_set():
                await self._beat()

    async def stop(self) -> None:
        """Signal the heartbeat to stop (wakes immediately)."""
        self._stop_event.set()
        logger.debug("Heartbeat service stopped")

    async def _beat(self) -> None:
        """Read HEARTBEAT.md and send to engine if it has content."""
        if not self._heartbeat_file.exists():
            logger.debug("No HEARTBEAT.md found, skipping")
            return

        content = self._heartbeat_file.read_text(encoding="utf-8").strip()
        if not content:
            logger.debug("HEARTBEAT.md is empty, skipping")
            return

        logger.info("Heartbeat triggered ({} chars)", len(content))
        try:
            result = await self._engine.run(content, session_key=SESSION_KEY)
            logger.info(
                "Heartbeat completed: {} iterations, {} tokens",
                result.iterations,
                result.total_tokens,
            )
            if self._reply_to and self._bus and result.response:
                await self._publish_result(result.response)
        except Exception as exc:
            logger.error("Heartbeat run failed: {}", exc)
            if self._reply_to and self._bus:
                await self._publish_result(f"Heartbeat run failed: {exc}")

    async def _publish_result(self, text: str) -> None:
        """Publish a heartbeat result to the message bus for channel delivery."""
        from grip.bus.events import OutboundMessage

        parts = self._reply_to.split(":", 1)
        if len(parts) != 2:
            logger.warning("Invalid reply_to format for heartbeat: {}", self._reply_to)
            return

        channel, chat_id = parts
        try:
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    text=text,
                )
            )
            logger.info("Heartbeat result published to {}:{}", channel, chat_id)
        except Exception as exc:
            logger.error("Failed to publish heartbeat result: {}", exc)
