"""Tests for the scheduler tool."""

from __future__ import annotations

import json

import pytest

from grip.tools.base import ToolContext
from grip.tools.scheduler import (
    SchedulerTool,
    create_scheduler_tools,
    parse_natural_language,
)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


class TestParseNaturalLanguage:
    def test_every_5_minutes(self):
        result = parse_natural_language("every 5 minutes")
        assert result == "*/5 * * * *"

    def test_every_minute(self):
        result = parse_natural_language("every minute")
        assert result == "* * * * *"

    def test_every_hour(self):
        result = parse_natural_language("every hour")
        assert result == "0 * * * *"

    def test_every_2_hours(self):
        result = parse_natural_language("every 2 hours")
        assert result == "0 */2 * * *"

    def test_every_day_at_9am(self):
        result = parse_natural_language("every day at 9am")
        assert result == "0 9 * * *"

    def test_every_day_at_9pm(self):
        result = parse_natural_language("every day at 9pm")
        assert result == "0 21 * * *"

    def test_every_day_at_14(self):
        result = parse_natural_language("every day at 14")
        assert result == "0 14 * * *"

    def test_every_monday_at_3pm(self):
        result = parse_natural_language("every Monday at 3pm")
        assert result == "0 15 * * 1"

    def test_every_friday_at_5pm(self):
        result = parse_natural_language("every Friday at 5pm")
        assert result == "0 17 * * 5"

    def test_every_month_on_the_1st(self):
        result = parse_natural_language("every month on the 1st")
        assert result == "0 0 1 * *"

    def test_every_month_on_the_15th(self):
        result = parse_natural_language("every month on the 15th")
        assert result == "0 0 15 * *"

    def test_every_weekday_at_9am(self):
        result = parse_natural_language("every weekday at 9am")
        assert result == "0 9 * * 1-5"

    def test_raw_cron_expression_passthrough(self):
        result = parse_natural_language("*/10 * * * *")
        assert result == "*/10 * * * *"

    def test_unrecognized_returns_none(self):
        result = parse_natural_language("whenever I feel like it")
        assert result is None

    def test_abbreviated_day_names(self):
        assert parse_natural_language("every Mon at 8am") == "0 8 * * 1"
        assert parse_natural_language("every Wed at 12pm") == "0 12 * * 3"
        assert parse_natural_language("every Sun at 6am") == "0 6 * * 0"


class TestSchedulerTool:
    def test_factory_returns_tool(self):
        tools = create_scheduler_tools()
        assert len(tools) == 1
        assert tools[0].name == "scheduler"

    @pytest.mark.asyncio
    async def test_create_action(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute({
            "action": "create",
            "schedule": "every 5 minutes",
            "task_name": "Health check",
            "command": "curl http://localhost/health",
        }, ctx)
        assert "Scheduled task created" in result
        assert "*/5 * * * *" in result

        cron_files = list((ctx.workspace_path / "cron").glob("*.json"))
        assert len(cron_files) == 1

    @pytest.mark.asyncio
    async def test_list_action_empty(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute({"action": "list"}, ctx)
        assert "No scheduled tasks" in result

    @pytest.mark.asyncio
    async def test_list_after_create(self, ctx):
        tool = SchedulerTool()
        await tool.execute({
            "action": "create",
            "schedule": "every hour",
            "task_name": "Backup",
            "command": "backup.sh",
        }, ctx)
        result = await tool.execute({"action": "list"}, ctx)
        assert "Backup" in result

    @pytest.mark.asyncio
    async def test_delete_action(self, ctx):
        tool = SchedulerTool()
        create_result = await tool.execute({
            "action": "create",
            "schedule": "every day at 9am",
            "task_name": "Report",
            "command": "generate_report.py",
        }, ctx)
        task_id = create_result.split("ID: ")[1].split("\n")[0].strip()

        delete_result = await tool.execute({
            "action": "delete",
            "task_id": task_id,
        }, ctx)
        assert "Deleted" in delete_result

        cron_files = list((ctx.workspace_path / "cron").glob("*.json"))
        assert len(cron_files) == 0

    @pytest.mark.asyncio
    async def test_invalid_schedule_returns_error(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute({
            "action": "create",
            "schedule": "whenever I feel like it",
            "task_name": "Random",
        }, ctx)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_error(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute({
            "action": "delete",
            "task_id": "nonexistent",
        }, ctx)
        assert "Error" in result
