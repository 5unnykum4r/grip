"""Tests for HISTORY.md rotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from grip.memory.manager import MemoryManager


@pytest.fixture
def memory_mgr(tmp_path: Path) -> MemoryManager:
    ws = tmp_path / "workspace"
    (ws / "memory").mkdir(parents=True)
    return MemoryManager(ws)


class TestHistoryRotation:
    def test_append_triggers_rotation_at_threshold(self, memory_mgr: MemoryManager):
        """When HISTORY.md exceeds threshold, rotation should happen."""
        memory_mgr._HISTORY_MAX_BYTES = 1000

        for i in range(50):
            memory_mgr.append_history(f"Entry {i}: " + "x" * 50)

        archives = list(memory_mgr._memory_dir.glob("HISTORY.archive.*.md"))
        assert len(archives) >= 1

    def test_rotation_preserves_recent_entries(self, memory_mgr: MemoryManager):
        """After rotation, HISTORY.md should contain recent entries."""
        memory_mgr._HISTORY_MAX_BYTES = 500

        for i in range(20):
            memory_mgr.append_history(f"Entry {i}")

        remaining = memory_mgr.read_history()
        assert "Entry 19" in remaining

    def test_rotation_creates_valid_archive(self, memory_mgr: MemoryManager):
        """Archive files should contain older entries."""
        memory_mgr._HISTORY_MAX_BYTES = 500

        for i in range(20):
            memory_mgr.append_history(f"Entry {i}: padding text here")

        archives = sorted(memory_mgr._memory_dir.glob("HISTORY.archive.*.md"))
        assert len(archives) >= 1
        all_archive_content = "".join(a.read_text(encoding="utf-8") for a in archives)
        assert "Entry" in all_archive_content

    def test_search_history_works_after_rotation(self, memory_mgr: MemoryManager):
        """search_history should work on remaining entries after rotation."""
        memory_mgr._HISTORY_MAX_BYTES = 500

        for i in range(20):
            memory_mgr.append_history(f"Entry {i}: unique keyword{i}")

        results = memory_mgr.search_history("keyword19")
        assert len(results) >= 1

    def test_no_rotation_below_threshold(self, memory_mgr: MemoryManager):
        """No rotation should happen when file is below threshold."""
        memory_mgr.append_history("Small entry")

        archives = list(memory_mgr._memory_dir.glob("HISTORY.archive.*.md"))
        assert len(archives) == 0
