"""Tests for the scheduler tool."""

from __future__ import annotations

import json

import pytest

from grip.tools.base import ToolContext
from grip.tools.scheduler import (
    SchedulerTool,
    _load_jobs_file,
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
    async def test_create_writes_to_jobs_json(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute(
            {
                "action": "create",
                "schedule": "every 5 minutes",
                "task_name": "Health check",
                "command": "curl http://localhost/health",
            },
            ctx,
        )
        assert "Scheduled task created" in result
        assert "*/5 * * * *" in result

        jobs_file = ctx.workspace_path / "cron" / "jobs.json"
        assert jobs_file.exists()
        jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
        assert len(jobs) == 1
        assert jobs[0]["schedule"] == "*/5 * * * *"
        assert jobs[0]["prompt"] == "curl http://localhost/health"
        assert jobs[0]["enabled"] is True
        assert jobs[0]["id"].startswith("cron_")

    @pytest.mark.asyncio
    async def test_create_has_cronjob_compatible_fields(self, ctx):
        tool = SchedulerTool()
        await tool.execute(
            {
                "action": "create",
                "schedule": "every hour",
                "task_name": "Test",
                "command": "echo hello",
                "reply_to": "telegram:12345",
            },
            ctx,
        )
        jobs = _load_jobs_file(ctx.workspace_path / "cron")
        job = jobs[0]
        assert "schedule" in job
        assert "prompt" in job
        assert "enabled" in job
        assert "last_run" in job
        assert "reply_to" in job
        assert job["reply_to"] == "telegram:12345"
        # Legacy field names must NOT be present
        assert "cron" not in job
        assert "command" not in job

    @pytest.mark.asyncio
    async def test_create_validates_reply_to_format(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute(
            {
                "action": "create",
                "schedule": "every hour",
                "task_name": "Test",
                "command": "echo hello",
                "reply_to": "bad_format",
            },
            ctx,
        )
        assert "Error" in result
        assert "reply_to" in result

    @pytest.mark.asyncio
    async def test_list_action_empty(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute({"action": "list"}, ctx)
        assert "No scheduled tasks" in result

    @pytest.mark.asyncio
    async def test_list_after_create(self, ctx):
        tool = SchedulerTool()
        await tool.execute(
            {
                "action": "create",
                "schedule": "every hour",
                "task_name": "Backup",
                "command": "backup.sh",
            },
            ctx,
        )
        result = await tool.execute({"action": "list"}, ctx)
        assert "Backup" in result
        assert "enabled" in result

    @pytest.mark.asyncio
    async def test_delete_action(self, ctx):
        tool = SchedulerTool()
        create_result = await tool.execute(
            {
                "action": "create",
                "schedule": "every day at 9am",
                "task_name": "Report",
                "command": "generate_report.py",
            },
            ctx,
        )
        task_id = create_result.split("ID: ")[1].split("\n")[0].strip()

        delete_result = await tool.execute(
            {"action": "delete", "task_id": task_id},
            ctx,
        )
        assert "Deleted" in delete_result

        jobs = _load_jobs_file(ctx.workspace_path / "cron")
        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_invalid_schedule_returns_error(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute(
            {
                "action": "create",
                "schedule": "whenever I feel like it",
                "task_name": "Random",
            },
            ctx,
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_error(self, ctx):
        tool = SchedulerTool()
        result = await tool.execute(
            {"action": "delete", "task_id": "nonexistent"},
            ctx,
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_migration_of_individual_files(self, ctx):
        """Old individual {id}.json files are migrated to jobs.json on execute()."""
        cron_dir = ctx.workspace_path / "cron"
        cron_dir.mkdir(parents=True, exist_ok=True)

        legacy_entry = {
            "id": "abc12345",
            "name": "Legacy Task",
            "cron": "*/10 * * * *",
            "command": "do_stuff.sh",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (cron_dir / "abc12345.json").write_text(json.dumps(legacy_entry), encoding="utf-8")

        tool = SchedulerTool()
        result = await tool.execute({"action": "list"}, ctx)
        assert "Legacy Task" in result

        # Individual file should be removed
        assert not (cron_dir / "abc12345.json").exists()

        # jobs.json should contain the migrated entry with correct field names
        jobs = _load_jobs_file(cron_dir)
        assert len(jobs) == 1
        assert jobs[0]["schedule"] == "*/10 * * * *"
        assert jobs[0]["prompt"] == "do_stuff.sh"
        assert jobs[0]["id"].startswith("cron_")
