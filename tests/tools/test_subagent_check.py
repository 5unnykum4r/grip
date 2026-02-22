"""Tests for CheckSubagentTool and ListSubagentsTool."""

from __future__ import annotations

import pytest

from grip.tools.base import ToolContext
from grip.tools.spawn import (
    CheckSubagentTool,
    ListSubagentsTool,
    SubagentInfo,
    SubagentManager,
)


@pytest.fixture
def manager() -> SubagentManager:
    mgr = SubagentManager()
    info = SubagentInfo(
        id="sub_test123", task_description="Test task", status="completed", result="Done!"
    )
    mgr._agents["sub_test123"] = info
    running = SubagentInfo(id="sub_running1", task_description="Running task", status="running")
    mgr._agents["sub_running1"] = running
    return mgr


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


class TestCheckSubagentTool:
    @pytest.mark.asyncio
    async def test_returns_completed_result(self, manager, ctx):
        tool = CheckSubagentTool(manager)
        result = await tool.execute({"agent_id": "sub_test123"}, ctx)
        assert "sub_test123" in result
        assert "completed" in result
        assert "Done!" in result

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error(self, manager, ctx):
        tool = CheckSubagentTool(manager)
        result = await tool.execute({"agent_id": "sub_nonexistent"}, ctx)
        assert "No subagent found" in result

    @pytest.mark.asyncio
    async def test_running_agent_shows_not_yet(self, manager, ctx):
        tool = CheckSubagentTool(manager)
        result = await tool.execute({"agent_id": "sub_running1"}, ctx)
        assert "running" in result
        assert "Not yet available" in result


class TestListSubagentsTool:
    @pytest.mark.asyncio
    async def test_lists_all_agents(self, manager, ctx):
        tool = ListSubagentsTool(manager)
        result = await tool.execute({}, ctx)
        assert "sub_test123" in result
        assert "sub_running1" in result

    @pytest.mark.asyncio
    async def test_empty_manager(self, ctx):
        tool = ListSubagentsTool(SubagentManager())
        result = await tool.execute({}, ctx)
        assert "No subagents" in result
