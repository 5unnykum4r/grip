"""Tests for CronService reload, locking, and HeartbeatService delivery."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grip.config.schema import CronConfig, HeartbeatConfig
from grip.cron.service import CronService
from grip.heartbeat.service import HeartbeatService


@dataclass
class FakeRunResult:
    response: str = "done"
    iterations: int = 1
    total_tokens: int = 100


class FakeEngine:
    """Minimal engine that satisfies EngineProtocol for testing."""

    def __init__(self, response: str = "done", delay: float = 0.0):
        self._response = response
        self._delay = delay
        self.run_count = 0

    async def run(self, prompt: str, session_key: str = "", **kwargs: Any) -> FakeRunResult:
        self.run_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return FakeRunResult(response=self._response)


class FakeBus:
    """Captures outbound messages for assertion."""

    def __init__(self):
        self.messages: list[Any] = []

    async def publish_outbound(self, msg: Any) -> None:
        self.messages.append(msg)


@pytest.fixture
def cron_dir(tmp_path) -> Path:
    d = tmp_path / "cron"
    d.mkdir()
    return d


@pytest.fixture
def config() -> CronConfig:
    return CronConfig(exec_timeout_minutes=1)


class TestCronJobPersistence:
    def test_add_and_list_jobs(self, cron_dir, config):
        engine = FakeEngine()
        svc = CronService(cron_dir, engine, config)
        job = svc.add_job("Test Job", "*/5 * * * *", "do something")
        assert job.id.startswith("cron_")
        assert len(svc.list_jobs()) == 1

        jobs_file = cron_dir / "jobs.json"
        assert jobs_file.exists()
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["schedule"] == "*/5 * * * *"

    def test_remove_job(self, cron_dir, config):
        engine = FakeEngine()
        svc = CronService(cron_dir, engine, config)
        job = svc.add_job("Remove Me", "0 * * * *", "bye")
        assert svc.remove_job(job.id) is True
        assert len(svc.list_jobs()) == 0

        data = json.loads((cron_dir / "jobs.json").read_text(encoding="utf-8"))
        assert len(data) == 0

    def test_remove_nonexistent_returns_false(self, cron_dir, config):
        engine = FakeEngine()
        svc = CronService(cron_dir, engine, config)
        assert svc.remove_job("cron_nonexistent") is False


class TestExternalJobReload:
    def test_load_jobs_picks_up_external_writes(self, cron_dir, config):
        engine = FakeEngine()
        svc = CronService(cron_dir, engine, config)
        assert len(svc.list_jobs()) == 0

        external_job = {
            "id": "cron_ext001",
            "name": "External Job",
            "schedule": "0 12 * * *",
            "prompt": "run external task",
            "enabled": True,
            "last_run": None,
            "created_at": datetime.now(UTC).isoformat(),
            "reply_to": "",
        }
        (cron_dir / "jobs.json").write_text(json.dumps([external_job], indent=2), encoding="utf-8")

        svc._load_jobs()
        assert len(svc.list_jobs()) == 1
        assert svc.get_job("cron_ext001") is not None
        assert svc.get_job("cron_ext001").name == "External Job"

    @pytest.mark.asyncio
    async def test_check_and_run_reloads_jobs(self, cron_dir, config):
        engine = FakeEngine()
        svc = CronService(cron_dir, engine, config)
        assert len(svc.list_jobs()) == 0

        external_job = {
            "id": "cron_reload1",
            "name": "Reload Test",
            "schedule": "* * * * *",
            "prompt": "hi",
            "enabled": True,
            "last_run": None,
            "created_at": "2020-01-01T00:00:00+00:00",
            "reply_to": "",
        }
        (cron_dir / "jobs.json").write_text(json.dumps([external_job], indent=2), encoding="utf-8")

        await svc._check_and_run_due_jobs()
        assert len(svc.list_jobs()) == 1


class TestJobLocking:
    @pytest.mark.asyncio
    async def test_executing_job_is_skipped(self, cron_dir, config):
        engine = FakeEngine(delay=0.5)
        svc = CronService(cron_dir, engine, config)
        job = svc.add_job("Slow Job", "* * * * *", "slow task")
        job.created_at = "2020-01-01T00:00:00+00:00"
        job.last_run = None
        svc._save_jobs()

        # First check fires the job
        await svc._check_and_run_due_jobs()
        # Yield to the event loop so the task starts and adds to _executing
        await asyncio.sleep(0.05)
        assert job.id in svc._executing

        # Second check while first is still running should skip the job
        await svc._check_and_run_due_jobs()
        # No new task should be added for the same job
        assert len([t for t in svc._pending_tasks if t.get_name() == f"cron-{job.id}"]) <= 1

        # Wait for execution to complete
        for task in list(svc._pending_tasks):
            await task

        # After completion, the job should be removed from _executing
        assert job.id not in svc._executing


class TestHeartbeatDelivery:
    @pytest.mark.asyncio
    async def test_heartbeat_publishes_to_bus(self, tmp_path):
        engine = FakeEngine(response="Heartbeat result")
        bus = FakeBus()
        hb_config = HeartbeatConfig(enabled=True, interval_minutes=5, reply_to="telegram:99999")

        (tmp_path / "HEARTBEAT.md").write_text("Check system health", encoding="utf-8")

        svc = HeartbeatService(
            tmp_path, engine, hb_config,
            bus=bus, reply_to=hb_config.reply_to,
        )
        await svc._beat()

        assert engine.run_count == 1
        assert len(bus.messages) == 1
        assert bus.messages[0].channel == "telegram"
        assert bus.messages[0].chat_id == "99999"
        assert bus.messages[0].text == "Heartbeat result"

    @pytest.mark.asyncio
    async def test_heartbeat_no_delivery_without_reply_to(self, tmp_path):
        engine = FakeEngine(response="No delivery")
        bus = FakeBus()
        hb_config = HeartbeatConfig(enabled=True, interval_minutes=5)

        (tmp_path / "HEARTBEAT.md").write_text("Check health", encoding="utf-8")

        svc = HeartbeatService(tmp_path, engine, hb_config, bus=bus, reply_to="")
        await svc._beat()

        assert engine.run_count == 1
        assert len(bus.messages) == 0

    @pytest.mark.asyncio
    async def test_heartbeat_publishes_on_failure(self, tmp_path):
        engine = MagicMock()
        engine.run = AsyncMock(side_effect=RuntimeError("engine broke"))
        bus = FakeBus()
        hb_config = HeartbeatConfig(enabled=True, interval_minutes=5, reply_to="discord:chan1")

        (tmp_path / "HEARTBEAT.md").write_text("Run checks", encoding="utf-8")

        svc = HeartbeatService(
            tmp_path, engine, hb_config,
            bus=bus, reply_to=hb_config.reply_to,
        )
        await svc._beat()

        assert len(bus.messages) == 1
        assert "failed" in bus.messages[0].text.lower()

    @pytest.mark.asyncio
    async def test_heartbeat_skip_empty_file(self, tmp_path):
        engine = FakeEngine()
        hb_config = HeartbeatConfig(enabled=True, interval_minutes=5)
        (tmp_path / "HEARTBEAT.md").write_text("", encoding="utf-8")

        svc = HeartbeatService(tmp_path, engine, hb_config)
        await svc._beat()

        assert engine.run_count == 0

    @pytest.mark.asyncio
    async def test_heartbeat_skip_missing_file(self, tmp_path):
        engine = FakeEngine()
        hb_config = HeartbeatConfig(enabled=True, interval_minutes=5)

        svc = HeartbeatService(tmp_path, engine, hb_config)
        await svc._beat()

        assert engine.run_count == 0
