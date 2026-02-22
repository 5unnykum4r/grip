"""Tests for TrackedEngine wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from grip.engines.tracked import TrackedEngine
from grip.engines.types import AgentRunResult, EngineProtocol
from grip.security.token_tracker import TokenLimitError, TokenTracker


@pytest.fixture
def mock_inner() -> MagicMock:
    inner = MagicMock(spec=EngineProtocol)
    inner.run = AsyncMock(
        return_value=AgentRunResult(
            response="hello",
            iterations=1,
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls_made=["read_file"],
        )
    )
    inner.consolidate_session = AsyncMock()
    inner.reset_session = AsyncMock()
    return inner


@pytest.fixture
def tracker(tmp_path: Path) -> TokenTracker:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return TokenTracker(state_dir, max_daily_tokens=10000)


class TestTrackedEngine:
    @pytest.mark.asyncio
    async def test_calls_check_limit_before_run(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        await tracked.run("hello")
        assert mock_inner.run.await_count == 1

    @pytest.mark.asyncio
    async def test_records_tokens_after_run(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        await tracked.run("hello")
        assert tracker.total_today == 150

    @pytest.mark.asyncio
    async def test_propagates_token_limit_error(self, mock_inner, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        small_tracker = TokenTracker(state_dir, max_daily_tokens=1)
        small_tracker.record(100, 100)

        tracked = TrackedEngine(mock_inner, small_tracker)
        with pytest.raises(TokenLimitError):
            await tracked.run("should fail")
        mock_inner.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_consolidate_session(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        await tracked.consolidate_session("test:session")
        mock_inner.consolidate_session.assert_awaited_once_with("test:session")

    @pytest.mark.asyncio
    async def test_delegates_reset_session(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        await tracked.reset_session("test:session")
        mock_inner.reset_session.assert_awaited_once_with("test:session")

    def test_is_engine_protocol(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        assert isinstance(tracked, EngineProtocol)

    def test_tracker_property(self, mock_inner, tracker):
        tracked = TrackedEngine(mock_inner, tracker)
        assert tracked.tracker is tracker
