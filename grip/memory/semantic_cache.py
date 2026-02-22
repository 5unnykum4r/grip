"""Semantic cache for LLM responses.

Stores LLM responses keyed by a hash of the user message + model name.
When the same (or sufficiently similar) query arrives within the TTL window,
the cached response is returned without making an LLM call — saving tokens
and latency.

Cache is stored on disk at workspace/state/semantic_cache.json with
configurable max entries and TTL. Entries expire based on creation time.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from loguru import logger


class SemanticCache:
    """Disk-backed LLM response cache with TTL expiry.

    Cache key is SHA-256 of (normalized_message + model). Responses are
    stored with timestamps and evicted when TTL expires or max_entries
    is exceeded (LRU by access time).
    """

    def __init__(
        self,
        state_dir: Path,
        *,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
        enabled: bool = True,
    ) -> None:
        self._state_dir = state_dir
        self._cache_file = state_dir / "semantic_cache.json"
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._enabled = enabled
        self._cache: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        """Load cache from disk, discarding expired entries."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        if not self._cache_file.exists():
            return {}
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            now = time.time()
            # Discard expired entries on load
            return {k: v for k, v in data.items() if now - v.get("created_at", 0) < self._ttl}
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt semantic cache file, resetting")
            return {}

    def _save(self) -> None:
        """Persist cache to disk atomically."""
        tmp = self._cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        tmp.rename(self._cache_file)

    @staticmethod
    def _make_key(message: str, model: str) -> str:
        """Generate a cache key from normalized message text + model name."""
        normalized = message.strip().lower()
        raw = f"{normalized}||{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, message: str, model: str) -> str | None:
        """Look up a cached response. Returns None on cache miss or if disabled."""
        if not self._enabled:
            return None

        key = self._make_key(message, model)
        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check TTL expiry
        if time.time() - entry.get("created_at", 0) >= self._ttl:
            del self._cache[key]
            return None

        # Update access time for LRU
        entry["accessed_at"] = time.time()
        logger.debug("Semantic cache hit for key {:.8}...", key)
        return entry.get("response")

    def put(self, message: str, model: str, response: str) -> None:
        """Store a response in the cache."""
        if not self._enabled:
            return

        key = self._make_key(message, model)
        now = time.time()
        self._cache[key] = {
            "response": response,
            "model": model,
            "created_at": now,
            "accessed_at": now,
            "message_preview": message[:100],
        }

        # Evict oldest entries if over capacity (LRU by accessed_at)
        if len(self._cache) > self._max_entries:
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("accessed_at", 0),
            )
            excess = len(self._cache) - self._max_entries
            for k in sorted_keys[:excess]:
                del self._cache[k]

        self._save()

    def invalidate(self, message: str, model: str) -> bool:
        """Remove a specific cache entry. Returns True if entry existed."""
        key = self._make_key(message, model)
        if key in self._cache:
            del self._cache[key]
            self._save()
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries. Returns the number of entries removed."""
        count = len(self._cache)
        self._cache.clear()
        self._save()
        return count

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        """Return cache statistics for status display."""
        now = time.time()
        active = sum(1 for v in self._cache.values() if now - v.get("created_at", 0) < self._ttl)
        return {
            "total_entries": len(self._cache),
            "active_entries": active,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
            "enabled": self._enabled,
        }
