"""Cron scheduling service: periodic task execution via cron expressions.

Jobs are stored as JSON in workspace/cron/jobs.json. When a job fires,
it calls EngineProtocol.run() with the job's prompt as the user message.

Uses croniter for cron expression parsing (e.g. "*/5 * * * *" = every 5 min).
Falls back to interval-based scheduling if croniter is not installed.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from grip.config.schema import CronConfig
from grip.engines.types import EngineProtocol


@dataclass(slots=True)
class CronJob:
    """A single scheduled task definition."""

    id: str
    name: str
    schedule: str
    prompt: str
    enabled: bool = True
    last_run: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reply_to: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CronJob:
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


class CronService:
    """Manages cron job persistence and periodic execution.

    On start(), spawns an asyncio loop that checks every 30 seconds
    whether any job is due. When a job fires, it calls engine.run()
    with the job's prompt as the user message. If the job has a reply_to
    session key (e.g. "telegram:12345"), the result is published to
    the message bus so it reaches the originating channel.
    """

    def __init__(
        self,
        cron_dir: Path,
        engine: EngineProtocol,
        config: CronConfig,
        bus: Any | None = None,
    ) -> None:
        self._cron_dir = cron_dir
        self._jobs_file = cron_dir / "jobs.json"
        self._engine = engine
        self._config = config
        self._bus = bus
        self._jobs: dict[str, CronJob] = {}
        self._running = False
        self._check_interval = 30

        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load jobs from the persistent JSON file."""
        if not self._jobs_file.exists():
            return
        try:
            data = json.loads(self._jobs_file.read_text(encoding="utf-8"))
            for item in data:
                job = CronJob.from_dict(item)
                self._jobs[job.id] = job
            logger.debug("Loaded {} cron jobs", len(self._jobs))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load cron jobs: {}", exc)

    def _save_jobs(self) -> None:
        """Persist all jobs to the JSON file atomically."""
        self._cron_dir.mkdir(parents=True, exist_ok=True)
        data = [job.to_dict() for job in self._jobs.values()]
        tmp = self._jobs_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(self._jobs_file)

    def add_job(self, name: str, schedule: str, prompt: str, reply_to: str = "") -> CronJob:
        """Create and persist a new cron job.

        Args:
            name: Human-readable job name.
            schedule: Cron expression (e.g. "*/5 * * * *").
            prompt: The prompt to send to the engine when the job fires.
            reply_to: Session key to route results to (e.g. "telegram:12345").
                      When set, job output is published to the message bus.

        Raises:
            ValueError: If reply_to is set but not in "channel:chat_id" format.
        """
        if reply_to and ":" not in reply_to:
            raise ValueError(
                f"Invalid reply_to format: '{reply_to}'. "
                "Expected 'channel:chat_id' (e.g. 'telegram:12345')."
            )
        job = CronJob(
            id=f"cron_{uuid.uuid4().hex[:8]}",
            name=name,
            schedule=schedule,
            prompt=prompt,
            reply_to=reply_to,
        )
        self._jobs[job.id] = job
        self._save_jobs()
        logger.info("Cron job added: {} ({})", name, schedule)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID. Returns True if found and removed."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save_jobs()
            return True
        return False

    def enable_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._save_jobs()
            return True
        return False

    def disable_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save_jobs()
            return True
        return False

    def list_jobs(self) -> list[CronJob]:
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> CronJob | None:
        return self._jobs.get(job_id)

    async def start(self) -> None:
        """Start the cron scheduler loop. Runs until cancelled."""
        self._running = True
        logger.info("Cron service started ({} jobs loaded)", len(self._jobs))

        while self._running:
            await self._check_and_run_due_jobs()
            await asyncio.sleep(self._check_interval)

    async def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False
        logger.info("Cron service stopped")

    async def _check_and_run_due_jobs(self) -> None:
        """Check all enabled jobs and run any that are due."""
        now = datetime.now(UTC)

        for job in self._jobs.values():
            if not job.enabled:
                continue

            if self._is_job_due(job, now):
                asyncio.create_task(
                    self._execute_job(job),
                    name=f"cron-{job.id}",
                )

    def _is_job_due(self, job: CronJob, now: datetime) -> bool:
        """Determine if a cron job should fire based on its schedule."""
        try:
            from croniter import croniter

            if job.last_run:
                last = datetime.fromisoformat(job.last_run)
            else:
                last = datetime.fromisoformat(job.created_at)
            cron = croniter(job.schedule, last)
            next_run = cron.get_next(datetime)
            return now >= next_run
        except ImportError:
            # Without croniter, parse simple interval expressions like "*/5 * * * *"
            if job.last_run:
                last = datetime.fromisoformat(job.last_run)
            else:
                last = datetime.fromisoformat(job.created_at)
            interval = self._parse_simple_interval(job.schedule)
            return (now - last).total_seconds() >= interval
        except Exception as exc:
            logger.error("Cron expression error for job {}: {}", job.id, exc)
            return False

    @staticmethod
    def _parse_simple_interval(schedule: str) -> float:
        """Extract minute interval from simple cron patterns like '*/N * * * *'.

        Returns interval in seconds. Defaults to 3600 (1 hour) if unparseable.
        """
        parts = schedule.strip().split()
        if parts and parts[0].startswith("*/"):
            try:
                minutes = int(parts[0][2:])
                return minutes * 60
            except ValueError:
                pass
        return 3600

    async def _execute_job(self, job: CronJob) -> None:
        """Run a single cron job through the engine with a timeout.

        If the job has a reply_to session key and a message bus is available,
        the result is published to the bus so the originating channel
        (e.g. Telegram) receives the response.
        """
        logger.info("Executing cron job: {} ({})", job.name, job.id)
        job.last_run = datetime.now(UTC).isoformat()
        self._save_jobs()

        timeout = self._config.exec_timeout_minutes * 60
        session_key = f"cron:{job.id}"

        try:
            result = await asyncio.wait_for(
                self._engine.run(job.prompt, session_key=session_key),
                timeout=timeout,
            )
            logger.info(
                "Cron job {} completed: {} iterations, response length {}",
                job.id,
                result.iterations,
                len(result.response),
            )

            # Route the result to the originating channel if reply_to is set
            if job.reply_to and self._bus and result.response:
                await self._publish_result(job, result.response)

        except TimeoutError:
            logger.error("Cron job {} timed out after {}s", job.id, timeout)
            if job.reply_to and self._bus:
                await self._publish_result(
                    job,
                    f"Cron job '{job.name}' timed out after {self._config.exec_timeout_minutes} minutes.",
                )
        except Exception as exc:
            logger.error("Cron job {} failed: {}", job.id, exc)
            if job.reply_to and self._bus:
                await self._publish_result(job, f"Cron job '{job.name}' failed: {exc}")

    async def _publish_result(self, job: CronJob, text: str) -> None:
        """Publish a cron job result to the message bus for channel delivery."""
        from grip.bus.events import OutboundMessage

        parts = job.reply_to.split(":", 1)
        if len(parts) != 2:
            logger.warning("Invalid reply_to format for cron job {}: {}", job.id, job.reply_to)
            return

        channel, chat_id = parts
        try:
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    text=text,
                )
            )
            logger.info("Cron job {} result published to {}:{}", job.id, channel, chat_id)
        except Exception as exc:
            logger.error("Failed to publish cron result for {}: {}", job.id, exc)
