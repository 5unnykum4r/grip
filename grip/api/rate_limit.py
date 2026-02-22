"""In-memory sliding-window rate limiter for the grip REST API.

Two instances are used in practice:
  - Per-IP limiter (applied before auth) to block brute-force attempts
  - Per-token limiter (applied after auth) to prevent accidental token drain

No external dependencies (Redis, etc.) — suitable for single-instance
self-hosted deployments.
"""

from __future__ import annotations

import time
from collections import defaultdict


class SlidingWindowRateLimiter:
    """Sliding-window counter rate limiter keyed by arbitrary string.

    Tracks request timestamps per key in a deque-like list. On each
    check, expired entries outside the window are pruned, then the
    current count is compared against the limit.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, int, float]:
        """Check if a request from `key` is allowed.

        Returns:
            (allowed, remaining, retry_after)
            - allowed: True if the request should proceed
            - remaining: number of requests left in the current window
            - retry_after: seconds until the oldest entry expires (0 if allowed)
        """
        now = time.monotonic()
        cutoff = now - self._window
        timestamps = self._requests[key]

        # Prune expired entries
        while timestamps and timestamps[0] <= cutoff:
            timestamps.pop(0)

        if len(timestamps) >= self._max_requests:
            retry_after = timestamps[0] + self._window - now
            return False, 0, max(retry_after, 0.1)

        timestamps.append(now)
        remaining = self._max_requests - len(timestamps)
        return True, remaining, 0.0

    def cleanup(self) -> int:
        """Remove keys with no recent requests. Returns number of keys removed."""
        now = time.monotonic()
        cutoff = now - self._window
        stale_keys = [
            key
            for key, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for key in stale_keys:
            del self._requests[key]
        return len(stale_keys)
