"""Tests for version consistency."""

from __future__ import annotations

import grip


class TestVersionConsistency:
    def test_version_is_0_2_0(self):
        assert grip.__version__ == "0.2.0"

    def test_version_is_string(self):
        assert isinstance(grip.__version__, str)

    def test_version_has_three_parts(self):
        parts = grip.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
