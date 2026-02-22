"""Channel manager: loads enabled channels from config and manages lifecycle.

On startup, instantiates all enabled channel implementations and starts
them concurrently. On shutdown, stops all channels gracefully.
"""

from __future__ import annotations

from loguru import logger

from grip.bus.queue import MessageBus
from grip.channels.base import BaseChannel
from grip.config.schema import ChannelsConfig


def _create_channel(name: str, config) -> BaseChannel | None:
    """Lazily import and instantiate a channel by name.

    Returns None if the optional dependency isn't installed.
    """
    try:
        if name == "telegram":
            from grip.channels.telegram import TelegramChannel

            return TelegramChannel(config)
        elif name == "discord":
            from grip.channels.discord import DiscordChannel

            return DiscordChannel(config)
        elif name == "slack":
            from grip.channels.slack import SlackChannel

            return SlackChannel(config)
        else:
            logger.warning("Unknown channel type: {}", name)
            return None
    except RuntimeError as exc:
        logger.error("Cannot load channel '{}': {}", name, exc)
        return None


class ChannelManager:
    """Manages the lifecycle of all enabled chat channels."""

    def __init__(self, channels_config: ChannelsConfig) -> None:
        self._config = channels_config
        self._channels: list[BaseChannel] = []

    @property
    def active_channels(self) -> list[BaseChannel]:
        return list(self._channels)

    async def start_all(self, bus: MessageBus) -> list[str]:
        """Start all enabled channels. Returns list of channel names started."""
        started: list[str] = []

        channel_entries = {
            "telegram": self._config.telegram,
            "discord": self._config.discord,
            "slack": self._config.slack,
        }

        for name, entry in channel_entries.items():
            if not entry.enabled:
                continue

            channel = _create_channel(name, entry)
            if channel is None:
                continue

            try:
                await channel.start(bus)
                self._channels.append(channel)
                started.append(name)
                logger.info("Channel '{}' started", name)
            except Exception as exc:
                logger.error("Failed to start channel '{}': {}", name, exc)

        return started

    async def stop_all(self) -> None:
        """Stop all running channels gracefully."""
        for channel in reversed(self._channels):
            try:
                await channel.stop()
                logger.info("Channel '{}' stopped", channel.name)
            except Exception as exc:
                logger.error("Error stopping channel '{}': {}", channel.name, exc)

        self._channels.clear()

    def get_channel(self, name: str) -> BaseChannel | None:
        """Look up a running channel by name."""
        for ch in self._channels:
            if ch.name == name:
                return ch
        return None
