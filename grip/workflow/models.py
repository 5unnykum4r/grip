"""Workflow data models: definitions, steps, and execution results.

A workflow is a DAG of steps. Each step runs an agent with a specific
profile and prompt. Steps can reference outputs of prior steps using
{{step_name.output}} template syntax. Independent steps (no dependency
edges between them) execute in parallel.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

_STEP_NAME_RE = re.compile(r"^[\w-]+$")


class StepStatus(StrEnum):
    """Lifecycle states for a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class StepDef:
    """Definition of a single workflow step.

    depends_on lists step names that must complete before this step runs.
    prompt can contain {{step_name.output}} placeholders that are resolved
    at execution time from prior step results.
    """

    name: str
    prompt: str
    profile: str = "default"
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass(slots=True)
class StepResult:
    """Execution result of a single workflow step."""

    name: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    iterations: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0

    def mark_running(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(UTC).isoformat()

    def mark_completed(self, output: str, iterations: int) -> None:
        self.status = StepStatus.COMPLETED
        self.output = output
        self.iterations = iterations
        self._set_completed_time()

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self._set_completed_time()

    def mark_skipped(self, reason: str) -> None:
        self.status = StepStatus.SKIPPED
        self.error = reason
        self._set_completed_time()

    def _set_completed_time(self) -> None:
        self.completed_at = datetime.now(UTC).isoformat()
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (end - start).total_seconds()


def _build_graph(steps: list[StepDef]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Build adjacency list and in-degree map from step definitions."""
    adj: dict[str, list[str]] = {s.name: [] for s in steps}
    in_degree: dict[str, int] = {s.name: 0 for s in steps}
    for step in steps:
        for dep in step.depends_on:
            adj[dep].append(step.name)
            in_degree[step.name] += 1
    return adj, in_degree


@dataclass(slots=True)
class WorkflowDef:
    """Complete workflow definition: a named DAG of steps.

    Steps are validated at load time to ensure:
      - Non-empty workflow name
      - At least one step
      - Valid step names (alphanumeric, underscore, hyphen)
      - Positive timeout values
      - No duplicate step names
      - All depends_on references point to existing steps
      - No circular dependencies
    """

    name: str
    description: str = ""
    steps: list[StepDef] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []

        if not self.name or not self.name.strip():
            errors.append("Workflow name cannot be empty")

        if not self.steps:
            errors.append("Workflow must have at least one step")
            return errors

        names = {s.name for s in self.steps}

        for step in self.steps:
            if not step.name or not _STEP_NAME_RE.match(step.name):
                errors.append(
                    f"Step name '{step.name}' is invalid "
                    "(must be non-empty, only alphanumeric/underscore/hyphen)"
                )
            if not step.prompt or not step.prompt.strip():
                errors.append(f"Step '{step.name}' has an empty prompt")
            if step.timeout_seconds < 1:
                errors.append(
                    f"Step '{step.name}' has invalid timeout ({step.timeout_seconds}s); must be >= 1"
                )

        if len(names) != len(self.steps):
            errors.append("Duplicate step names found")

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        if not errors:
            layers = self.get_execution_order()
            total_in_layers = sum(len(layer) for layer in layers)
            if total_in_layers != len(self.steps):
                errors.append("Circular dependency detected in workflow steps")

        return errors

    def get_execution_order(self) -> list[list[str]]:
        """Return steps grouped into parallel execution layers.

        Each layer contains steps whose dependencies are all in earlier
        layers, so they can execute concurrently.
        """
        adj, in_degree = _build_graph(self.steps)
        layers: list[list[str]] = []
        queue = deque(n for n, d in in_degree.items() if d == 0)

        while queue:
            layer = sorted(queue)
            layers.append(layer)
            queue.clear()
            for node in layer:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": s.name,
                    "prompt": s.prompt,
                    "profile": s.profile,
                    "depends_on": s.depends_on,
                    "timeout_seconds": s.timeout_seconds,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDef:
        steps = [
            StepDef(
                name=s["name"],
                prompt=s["prompt"],
                profile=s.get("profile", "default"),
                depends_on=s.get("depends_on", []),
                timeout_seconds=int(s.get("timeout_seconds", 300)),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
        )


_RESULT_OUTPUT_LIMIT = 500


@dataclass(slots=True)
class WorkflowRunResult:
    """Complete result of a workflow execution."""

    workflow_name: str
    status: str = "pending"
    step_results: dict[str, StepResult] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0

    @property
    def all_completed(self) -> bool:
        return all(r.status == StepStatus.COMPLETED for r in self.step_results.values())

    @property
    def has_failures(self) -> bool:
        return any(r.status == StepStatus.FAILED for r in self.step_results.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "steps": {
                name: {
                    "status": r.status.value,
                    "output": (
                        r.output[:_RESULT_OUTPUT_LIMIT] + "... [truncated]"
                        if len(r.output) > _RESULT_OUTPUT_LIMIT
                        else r.output
                    ),
                    "error": r.error,
                    "iterations": r.iterations,
                    "duration_seconds": r.duration_seconds,
                }
                for name, r in self.step_results.items()
            },
        }
