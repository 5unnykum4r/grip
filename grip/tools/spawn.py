"""Spawn tool: create background subagents for async task execution.

Subagents run in independent asyncio tasks with their own message
history. They share the same LLM provider and tool registry but
have isolated context. Results are reported back via the message tool.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext


@dataclass(slots=True)
class SubagentInfo:
    """Tracking record for a spawned subagent."""

    id: str
    task_description: str
    status: str = "running"
    result: str | None = None
    asyncio_task: asyncio.Task[Any] | None = field(default=None, repr=False)


class SubagentManager:
    """Manages the lifecycle of background subagents.

    Subagents are spawned as asyncio tasks and tracked by ID. When a
    subagent finishes, its result is stored and can be retrieved.
    """

    def __init__(self) -> None:
        self._agents: dict[str, SubagentInfo] = {}

    def spawn(
        self,
        task_description: str,
        run_coro: Any,
    ) -> SubagentInfo:
        """Create a new subagent and schedule it as an asyncio task.

        Args:
            task_description: What the subagent should do.
            run_coro: A coroutine that executes the subagent's work.

        Returns:
            SubagentInfo with the assigned ID.
        """
        agent_id = f"sub_{uuid.uuid4().hex[:8]}"
        info = SubagentInfo(id=agent_id, task_description=task_description)

        async def _wrapper() -> None:
            try:
                result = await run_coro
                info.result = result
                info.status = "completed"
                logger.info("Subagent {} completed", agent_id)
            except asyncio.CancelledError:
                info.status = "cancelled"
                logger.info("Subagent {} cancelled", agent_id)
            except Exception as exc:
                info.result = f"Error: {exc}"
                info.status = "failed"
                logger.error("Subagent {} failed: {}", agent_id, exc)

        info.asyncio_task = asyncio.create_task(_wrapper(), name=f"subagent-{agent_id}")
        self._agents[agent_id] = info
        logger.info("Spawned subagent {}: {}", agent_id, task_description[:100])
        return info

    def get(self, agent_id: str) -> SubagentInfo | None:
        return self._agents.get(agent_id)

    def list_active(self) -> list[SubagentInfo]:
        return [a for a in self._agents.values() if a.status == "running"]

    def list_all(self) -> list[SubagentInfo]:
        return list(self._agents.values())

    async def cancel(self, agent_id: str) -> bool:
        info = self._agents.get(agent_id)
        if info and info.asyncio_task and not info.asyncio_task.done():
            info.asyncio_task.cancel()
            return True
        return False

    async def cancel_all(self) -> int:
        count = 0
        for info in self._agents.values():
            if info.asyncio_task and not info.asyncio_task.done():
                info.asyncio_task.cancel()
                count += 1
        return count


class SpawnTool(Tool):
    """Tool that creates background subagents.

    The actual agent execution is wired up externally via the spawn_callback.
    When the LLM calls this tool, it creates a subagent entry and delegates
    execution to the callback which runs the agent loop with the given task.
    """

    def __init__(self, subagent_manager: SubagentManager | None = None) -> None:
        self._manager = subagent_manager or SubagentManager()

    @property
    def category(self) -> str:
        return "orchestration"

    @property
    def manager(self) -> SubagentManager:
        return self._manager

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a background subagent for a task. Use for long-running or parallel work."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task for the subagent to perform.",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or instructions for the subagent.",
                },
            },
            "required": ["task"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        task_desc = params["task"]
        extra_context = params.get("context", "")

        spawn_callback = ctx.extra.get("spawn_callback")
        if spawn_callback is None:
            return (
                "Error: Subagent spawning is not available in this mode. "
                "Try running the task directly instead."
            )

        full_task = task_desc
        if extra_context:
            full_task = f"{task_desc}\n\nAdditional context: {extra_context}"

        coro = spawn_callback(full_task, ctx.session_key)
        info = self._manager.spawn(task_desc, coro)

        return (
            f"Subagent spawned: {info.id}\n"
            f"Task: {task_desc[:200]}\n"
            f"Status: {info.status}\n"
            f"The subagent will report results via send_message when done."
        )


class CheckSubagentTool(Tool):
    """Check the status and result of a previously spawned subagent."""

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "check_subagent"

    @property
    def description(self) -> str:
        return "Check the status and result of a previously spawned subagent by its ID."

    @property
    def category(self) -> str:
        return "orchestration"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the subagent to check.",
                },
            },
            "required": ["agent_id"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        info = self._manager.get(params["agent_id"])
        if not info:
            return (
                f"No subagent found with ID '{params['agent_id']}'. Use list_subagents to see all."
            )
        return (
            f"ID: {info.id}\n"
            f"Task: {info.task_description}\n"
            f"Status: {info.status}\n"
            f"Result: {info.result or 'Not yet available'}"
        )


class ListSubagentsTool(Tool):
    """List all spawned subagents and their statuses."""

    def __init__(self, manager: SubagentManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "list_subagents"

    @property
    def description(self) -> str:
        return "List all spawned subagents with their IDs, statuses, and task descriptions."

    @property
    def category(self) -> str:
        return "orchestration"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        agents = self._manager.list_all()
        if not agents:
            return "No subagents have been spawned."
        lines = []
        for a in agents:
            lines.append(f"- {a.id} [{a.status}]: {a.task_description[:80]}")
        return "\n".join(lines)


def create_spawn_tools(manager: SubagentManager | None = None) -> list[Tool]:
    mgr = manager or SubagentManager()
    return [SpawnTool(mgr), CheckSubagentTool(mgr), ListSubagentsTool(mgr)]
