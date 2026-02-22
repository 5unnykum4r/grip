"""Persistent knowledge base for structured learnings.

Goes beyond MEMORY.md by storing typed knowledge entries in a JSON-backed
store. Each entry has a category, content, source, and timestamp. The
knowledge base supports search by category and keyword, deduplication
of similar entries, and automatic expiry of stale entries.

Categories:
  - user_preference: User's stated preferences and habits
  - project_decision: Technical decisions made during development
  - system_behavior: Observed behaviors of tools, APIs, or services
  - learned_fact: General facts discovered during interactions
  - error_pattern: Recurring error patterns and their solutions
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass(slots=True)
class KnowledgeEntry:
    """A single knowledge base entry."""

    id: str
    category: str
    content: str
    source: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0


VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "user_preference",
        "project_decision",
        "system_behavior",
        "learned_fact",
        "error_pattern",
    }
)


def _make_id(category: str, content: str) -> str:
    """Generate a deterministic ID from category + content for dedup."""
    raw = f"{category}:{content.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class KnowledgeBase:
    """JSON-backed persistent knowledge store.

    Stored at workspace/memory/knowledge.json. Supports typed entries,
    keyword search, category filtering, and automatic deduplication.
    """

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = memory_dir
        self._kb_path = memory_dir / "knowledge.json"
        self._entries: dict[str, KnowledgeEntry] = self._load()

    def _load(self) -> dict[str, KnowledgeEntry]:
        """Load knowledge entries from disk."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        if not self._kb_path.exists():
            return {}
        try:
            data = json.loads(self._kb_path.read_text(encoding="utf-8"))
            entries: dict[str, KnowledgeEntry] = {}
            for entry_data in data.get("entries", []):
                entry = KnowledgeEntry(
                    id=entry_data["id"],
                    category=entry_data["category"],
                    content=entry_data["content"],
                    source=entry_data.get("source", ""),
                    tags=entry_data.get("tags", []),
                    created_at=entry_data.get("created_at", time.time()),
                    accessed_at=entry_data.get("accessed_at", time.time()),
                    access_count=entry_data.get("access_count", 0),
                )
                entries[entry.id] = entry
            return entries
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupt knowledge base file, starting fresh: {}", exc)
            return {}

    def _save(self) -> None:
        """Persist knowledge base to disk atomically."""
        data = {
            "version": 1,
            "entries": [asdict(e) for e in self._entries.values()],
        }
        tmp = self._kb_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.rename(self._kb_path)

    def add(
        self,
        category: str,
        content: str,
        *,
        source: str = "",
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """Add a knowledge entry. Deduplicates by content hash within category.

        If an entry with the same category+content hash already exists,
        updates its access time and returns the existing entry.
        """
        if category not in VALID_CATEGORIES:
            logger.warning(
                "Unknown knowledge category '{}', defaulting to 'learned_fact'", category
            )
            category = "learned_fact"

        entry_id = _make_id(category, content)

        if entry_id in self._entries:
            existing = self._entries[entry_id]
            existing.accessed_at = time.time()
            existing.access_count += 1
            self._save()
            logger.debug("Knowledge entry already exists (id={}), updated access time", entry_id)
            return existing

        entry = KnowledgeEntry(
            id=entry_id,
            category=category,
            content=content.strip(),
            source=source,
            tags=tags or [],
        )
        self._entries[entry_id] = entry
        self._save()
        logger.info("Added knowledge entry: {} (category={})", entry_id, category)
        return entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Retrieve a single entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.accessed_at = time.time()
            entry.access_count += 1
        return entry

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        max_results: int = 20,
    ) -> list[KnowledgeEntry]:
        """Search knowledge entries by keyword and/or category.

        Returns entries sorted by relevance (access count + recency).
        """
        results: list[KnowledgeEntry] = []
        query_lower = query.lower()

        for entry in self._entries.values():
            if category and entry.category != category:
                continue
            if query_lower:
                searchable = f"{entry.content} {' '.join(entry.tags)} {entry.source}".lower()
                if query_lower not in searchable:
                    continue
            results.append(entry)

        # Sort by access_count (descending), then by created_at (descending)
        results.sort(key=lambda e: (e.access_count, e.created_at), reverse=True)
        return results[:max_results]

    def by_category(self, category: str) -> list[KnowledgeEntry]:
        """Get all entries in a specific category, sorted by creation time."""
        entries = [e for e in self._entries.values() if e.category == category]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if it existed."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    def clear_category(self, category: str) -> int:
        """Remove all entries in a category. Returns count removed."""
        to_remove = [eid for eid, e in self._entries.items() if e.category == category]
        for eid in to_remove:
            del self._entries[eid]
        if to_remove:
            self._save()
        return len(to_remove)

    @property
    def count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        category_counts: dict[str, int] = {}
        for entry in self._entries.values():
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        return {
            "total_entries": len(self._entries),
            "categories": category_counts,
        }

    def export_for_context(self, *, max_chars: int = 2000) -> str:
        """Export the most relevant entries as a text block for LLM context.

        Prioritizes frequently accessed entries and user preferences.
        Truncates to max_chars to fit within context budgets.
        """
        # Prioritize user preferences, then by access count
        priority_order = [
            "user_preference",
            "project_decision",
            "error_pattern",
            "system_behavior",
            "learned_fact",
        ]
        sorted_entries: list[KnowledgeEntry] = []
        for cat in priority_order:
            cat_entries = self.by_category(cat)
            sorted_entries.extend(cat_entries)

        lines: list[str] = []
        total_chars = 0
        for entry in sorted_entries:
            line = f"[{entry.category}] {entry.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(lines)
