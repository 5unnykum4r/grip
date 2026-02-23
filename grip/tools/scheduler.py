"""Scheduler tool — natural language to cron expression conversion.

Rule-based parser (no LLM call) that converts human-readable scheduling
phrases into standard cron expressions. Actions: create, list, delete.

Integrates with grip's workspace by writing cron entries to the
``cron/`` directory.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grip.tools.base import Tool, ToolContext

_DAY_MAP = {
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
    "sunday": "0",
    "mon": "1",
    "tue": "2",
    "wed": "3",
    "thu": "4",
    "fri": "5",
    "sat": "6",
    "sun": "0",
}

_NL_PATTERNS: list[tuple[re.Pattern[str], str | callable]] = [
    (re.compile(r"every\s+(\d+)\s+minutes?", re.IGNORECASE), lambda m: f"*/{m.group(1)} * * * *"),
    (re.compile(r"every\s+(\d+)\s+hours?", re.IGNORECASE), lambda m: f"0 */{m.group(1)} * * *"),
    (re.compile(r"every\s+minute", re.IGNORECASE), lambda m: "* * * * *"),
    (re.compile(r"every\s+hour", re.IGNORECASE), lambda m: "0 * * * *"),
    (re.compile(r"every\s+day\s+at\s+(\d{1,2})\s*(am|pm)?", re.IGNORECASE), None),
    (
        re.compile(
            r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
            r"\s+at\s+(\d{1,2})\s*(am|pm)?",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        re.compile(r"every\s+month\s+on\s+the\s+(\d{1,2})(st|nd|rd|th)?", re.IGNORECASE),
        lambda m: f"0 0 {m.group(1)} * *",
    ),
    (re.compile(r"every\s+weekday\s+at\s+(\d{1,2})\s*(am|pm)?", re.IGNORECASE), None),
]


def _parse_hour(hour_str: str, ampm: str | None) -> int:
    """Convert 12-hour or 24-hour time string to 24-hour integer."""
    hour = int(hour_str)
    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
    return hour


def parse_natural_language(expression: str) -> str | None:
    """Convert a natural language scheduling phrase to a cron expression.

    Returns the cron string or None if the phrase is not recognized.
    """
    text = expression.strip()

    for pattern, handler in _NL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if callable(handler):
            return handler(match)

    every_day_match = re.search(r"every\s+day\s+at\s+(\d{1,2})\s*(am|pm)?", text, re.IGNORECASE)
    if every_day_match:
        hour = _parse_hour(every_day_match.group(1), every_day_match.group(2))
        return f"0 {hour} * * *"

    day_match = re.search(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
        r"\s+at\s+(\d{1,2})\s*(am|pm)?",
        text,
        re.IGNORECASE,
    )
    if day_match:
        day = _DAY_MAP.get(day_match.group(1).lower(), "0")
        hour = _parse_hour(day_match.group(2), day_match.group(3))
        return f"0 {hour} * * {day}"

    weekday_match = re.search(r"every\s+weekday\s+at\s+(\d{1,2})\s*(am|pm)?", text, re.IGNORECASE)
    if weekday_match:
        hour = _parse_hour(weekday_match.group(1), weekday_match.group(2))
        return f"0 {hour} * * 1-5"

    cron_match = re.match(
        r"^([*\d/,\-]+\s+[*\d/,\-]+\s+[*\d/,\-]+\s+[*\d/,\-]+\s+[*\d/,\-]+)$", text
    )
    if cron_match:
        return cron_match.group(1)

    return None


def _load_jobs_file(cron_dir: Path) -> list[dict[str, Any]]:
    """Load jobs from workspace/cron/jobs.json. Returns empty list on missing/corrupt file."""
    jobs_file = cron_dir / "jobs.json"
    if not jobs_file.exists():
        return []
    try:
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_jobs_file(cron_dir: Path, jobs: list[dict[str, Any]]) -> None:
    """Persist jobs list to workspace/cron/jobs.json atomically."""
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = cron_dir / "jobs.json"
    tmp = jobs_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    tmp.rename(jobs_file)


def _migrate_individual_files(cron_dir: Path) -> None:
    """One-time migration: convert old {id}.json files into jobs.json entries.

    Converts legacy field names (cron -> schedule, command -> prompt) to
    CronJob-compatible format. Already-migrated files are skipped.
    """
    individual_files = [
        f for f in cron_dir.glob("*.json")
        if f.name != "jobs.json" and not f.name.endswith(".tmp")
    ]
    if not individual_files:
        return

    jobs = _load_jobs_file(cron_dir)
    existing_ids = {j.get("id") for j in jobs}

    for f in individual_files:
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        entry_id = entry.get("id", "")
        if entry_id in existing_ids:
            f.unlink()
            continue

        migrated: dict[str, Any] = {
            "id": f"cron_{entry_id}" if not entry_id.startswith("cron_") else entry_id,
            "name": entry.get("name", "Unnamed task"),
            "schedule": entry.get("cron", entry.get("schedule", "")),
            "prompt": entry.get("command", entry.get("prompt", "")),
            "enabled": entry.get("enabled", True),
            "last_run": entry.get("last_run", None),
            "created_at": entry.get("created_at", datetime.now(UTC).isoformat()),
            "reply_to": entry.get("reply_to", ""),
        }
        jobs.append(migrated)
        f.unlink()

    _save_jobs_file(cron_dir, jobs)


class SchedulerTool(Tool):
    """Natural language cron scheduling: create, list, and delete tasks."""

    @property
    def name(self) -> str:
        return "scheduler"

    @property
    def description(self) -> str:
        return (
            "Manage scheduled tasks with natural language ('every day at 9am') or cron expressions."
        )

    @property
    def category(self) -> str:
        return "general"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "delete"],
                    "description": "Action to perform.",
                },
                "schedule": {
                    "type": "string",
                    "description": "Natural language or cron expression (for create action).",
                },
                "task_name": {
                    "type": "string",
                    "description": "Name/description of the scheduled task (for create action).",
                },
                "command": {
                    "type": "string",
                    "description": "Command or message to execute on schedule (for create action).",
                },
                "reply_to": {
                    "type": "string",
                    "description": "Session key to deliver results to (e.g. 'telegram:12345'). Required for channel delivery.",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID to delete (for delete action).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        action = params.get("action", "")
        cron_dir = ctx.workspace_path / "cron"
        cron_dir.mkdir(parents=True, exist_ok=True)

        _migrate_individual_files(cron_dir)

        if action == "create":
            return self._create(params, cron_dir)
        elif action == "list":
            return self._list(cron_dir)
        elif action == "delete":
            return self._delete(params, cron_dir)
        else:
            return f"Error: unknown action '{action}'. Use: create, list, delete."

    def _create(self, params: dict[str, Any], cron_dir: Path) -> str:
        schedule = params.get("schedule", "")
        task_name = params.get("task_name", "Unnamed task")
        command = params.get("command", "")
        reply_to = params.get("reply_to", "")

        if not schedule:
            return "Error: schedule is required for create action."

        if reply_to and ":" not in reply_to:
            return (
                f"Error: invalid reply_to format '{reply_to}'. "
                "Expected 'channel:chat_id' (e.g. 'telegram:12345')."
            )

        cron_expr = parse_natural_language(schedule)
        if cron_expr is None:
            return (
                f"Error: could not parse schedule '{schedule}'. "
                "Try formats like: 'every 5 minutes', 'every day at 9am', 'every Monday at 3pm', "
                "or a raw cron expression like '*/5 * * * *'."
            )

        task_id = f"cron_{uuid.uuid4().hex[:8]}"
        entry: dict[str, Any] = {
            "id": task_id,
            "name": task_name,
            "schedule": cron_expr,
            "prompt": command,
            "enabled": True,
            "last_run": None,
            "created_at": datetime.now(UTC).isoformat(),
            "reply_to": reply_to,
        }

        jobs = _load_jobs_file(cron_dir)
        jobs.append(entry)
        _save_jobs_file(cron_dir, jobs)

        result = (
            f"Scheduled task created:\n"
            f"  ID: {task_id}\n"
            f"  Name: {task_name}\n"
            f"  Cron: {cron_expr}\n"
            f"  Schedule: {schedule}\n"
            f"  Prompt: {command}"
        )
        if reply_to:
            result += f"\n  Reply to: {reply_to}"
        return result

    def _list(self, cron_dir: Path) -> str:
        jobs = _load_jobs_file(cron_dir)
        if not jobs:
            return "No scheduled tasks found."

        lines = ["## Scheduled Tasks\n"]
        for entry in jobs:
            enabled = "enabled" if entry.get("enabled", True) else "disabled"
            lines.append(
                f"- **{entry.get('name', 'Unnamed')}** (ID: {entry.get('id', '?')}) [{enabled}]\n"
                f"  Schedule: `{entry.get('schedule', 'N/A')}` | Prompt: {entry.get('prompt', 'N/A')}"
            )
        return "\n".join(lines)

    def _delete(self, params: dict[str, Any], cron_dir: Path) -> str:
        task_id = params.get("task_id", "")
        if not task_id:
            return "Error: task_id is required for delete action."

        jobs = _load_jobs_file(cron_dir)
        filtered = [j for j in jobs if j.get("id") != task_id]

        if len(filtered) == len(jobs):
            return f"Error: no scheduled task found with ID '{task_id}'."

        _save_jobs_file(cron_dir, filtered)
        return f"Deleted scheduled task: {task_id}"


def create_scheduler_tools() -> list[Tool]:
    """Factory function returning scheduler tool instances."""
    return [SchedulerTool()]
