"""Daily token usage tracking with configurable limits.

Persists daily token counts to workspace/state/token_usage.json.
Resets automatically at midnight UTC. When max_daily_tokens is
configured and exceeded, raises TokenLimitError.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


class TokenLimitError(Exception):
    """Raised when the daily token limit is reached."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(
            f"Daily token limit exceeded: {used:,} used of {limit:,} allowed. "
            f"Resets at midnight UTC. Adjust agents.defaults.max_daily_tokens in config."
        )


class TokenTracker:
    """Tracks daily token usage and enforces limits.

    Usage file format (workspace/state/token_usage.json):
    {
        "date": "2026-02-21",
        "prompt_tokens": 12345,
        "completion_tokens": 6789,
        "total_tokens": 19134,
        "request_count": 42
    }
    """

    def __init__(self, state_dir: Path, max_daily_tokens: int = 0) -> None:
        self._state_dir = state_dir
        self._usage_file = state_dir / "token_usage.json"
        self._max_daily = max_daily_tokens
        self._lock = threading.Lock()
        self._data = self._load()

    def _today(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        """Load usage data, resetting if the date has changed."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        if self._usage_file.exists():
            try:
                data = json.loads(self._usage_file.read_text(encoding="utf-8"))
                if data.get("date") == self._today():
                    return data
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt token usage file, resetting")
        return self._empty()

    def _empty(self) -> dict:
        return {
            "date": self._today(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }

    def _save(self) -> None:
        tmp = self._usage_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.rename(self._usage_file)

    def check_limit(self) -> None:
        """Raise TokenLimitError if the daily limit would be exceeded.

        Call this BEFORE making an LLM request.
        """
        with self._lock:
            if self._data.get("date") != self._today():
                self._data = self._empty()

            if self._max_daily > 0 and self._data["total_tokens"] >= self._max_daily:
                raise TokenLimitError(self._data["total_tokens"], self._max_daily)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record tokens from a completed LLM call (thread-safe)."""
        with self._lock:
            if self._data.get("date") != self._today():
                self._data = self._empty()

            self._data["prompt_tokens"] += prompt_tokens
            self._data["completion_tokens"] += completion_tokens
            self._data["total_tokens"] += prompt_tokens + completion_tokens
            self._data["request_count"] += 1
            self._save()

        if self._max_daily > 0:
            remaining = self._max_daily - self._data["total_tokens"]
            if remaining < self._max_daily * 0.1:
                logger.warning(
                    "Token budget low: {:,} / {:,} used ({:,} remaining)",
                    self._data["total_tokens"],
                    self._max_daily,
                    max(0, remaining),
                )

    @property
    def total_today(self) -> int:
        if self._data.get("date") != self._today():
            return 0
        return self._data["total_tokens"]

    @property
    def requests_today(self) -> int:
        if self._data.get("date") != self._today():
            return 0
        return self._data["request_count"]

    @property
    def remaining(self) -> int | None:
        """Remaining tokens today, or None if unlimited."""
        if self._max_daily <= 0:
            return None
        return max(0, self._max_daily - self.total_today)

    def summary(self) -> dict:
        """Return a summary dict for status display."""
        return {
            "date": self._data.get("date", self._today()),
            "total_tokens": self.total_today,
            "prompt_tokens": self._data.get("prompt_tokens", 0),
            "completion_tokens": self._data.get("completion_tokens", 0),
            "request_count": self.requests_today,
            "limit": self._max_daily if self._max_daily > 0 else "unlimited",
            "remaining": self.remaining,
        }
