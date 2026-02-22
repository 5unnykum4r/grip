"""Workflow execution engine: runs DAG-based multi-agent workflows.

The engine takes a WorkflowDef and executes its steps respecting
dependency order. Independent steps within the same layer run
concurrently via asyncio.gather. Step prompts can reference prior
step outputs using {{step_name.output}} template syntax.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

from loguru import logger

from grip.config.schema import AgentProfile, GripConfig
from grip.engines.types import EngineProtocol
from grip.tools.base import ToolRegistry
from grip.workflow.models import (
    StepDef,
    StepResult,
    StepStatus,
    WorkflowDef,
    WorkflowRunResult,
)

TEMPLATE_PATTERN = re.compile(r"\{\{(\w+)\.output\}\}")


class WorkflowEngine:
    """Executes multi-step workflows using agent profiles.

    Each step runs through an EngineProtocol implementation configured per
    the step's profile. The engine manages step ordering (parallel layers),
    template resolution, and result aggregation.
    """

    def __init__(
        self,
        config: GripConfig,
        engine: EngineProtocol,
        tool_registry: ToolRegistry,
    ) -> None:
        self._config = config
        self._engine = engine
        self._registry = tool_registry
        self._profiles = config.agents.profiles

    async def run(self, workflow: WorkflowDef) -> WorkflowRunResult:
        """Execute a workflow end-to-end, returning the aggregated result."""
        errors = workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        layers = workflow.get_execution_order()
        result = WorkflowRunResult(
            workflow_name=workflow.name,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )

        step_map = {s.name: s for s in workflow.steps}
        for step in workflow.steps:
            result.step_results[step.name] = StepResult(name=step.name)

        logger.info(
            "Workflow '{}' starting: {} steps in {} layers",
            workflow.name,
            len(workflow.steps),
            len(layers),
        )

        for layer_idx, layer_names in enumerate(layers, 1):
            logger.info("Executing layer {}/{}: {}", layer_idx, len(layers), layer_names)

            tasks = []
            for step_name in layer_names:
                step_def = step_map[step_name]
                step_result = result.step_results[step_name]
                resolved_prompt = self._resolve_template(step_def.prompt, result.step_results)
                tasks.append(self._execute_step(step_def, step_result, resolved_prompt))

            await asyncio.gather(*tasks)

            if any(result.step_results[name].status == StepStatus.FAILED for name in layer_names):
                logger.warning("Layer {} had failures, skipping dependent steps", layer_idx)
                self._skip_dependents(layer_names, layers[layer_idx:], result, step_map)
                break

        result.completed_at = datetime.now(UTC).isoformat()
        if result.has_failures:
            result.status = "failed"
        elif result.all_completed:
            result.status = "completed"
        else:
            result.status = "partial"

        start = datetime.fromisoformat(result.started_at)
        end = datetime.fromisoformat(result.completed_at)
        result.total_duration_seconds = (end - start).total_seconds()

        logger.info(
            "Workflow '{}' {}: {:.1f}s",
            workflow.name,
            result.status,
            result.total_duration_seconds,
        )
        return result

    async def _execute_step(
        self,
        step_def: StepDef,
        step_result: StepResult,
        resolved_prompt: str,
    ) -> None:
        """Execute a single workflow step through the engine."""
        step_result.mark_running()
        profile = self._profiles.get(step_def.profile, AgentProfile())

        model_override = profile.model if profile.model else None
        session_key = f"workflow:{step_def.name}"

        try:
            agent_result = await asyncio.wait_for(
                self._engine.run(
                    resolved_prompt,
                    session_key=session_key,
                    model=model_override,
                ),
                timeout=step_def.timeout_seconds,
            )
            step_result.mark_completed(agent_result.response, agent_result.iterations)
            logger.info(
                "Step '{}' completed: {} iterations, {:.1f}s",
                step_def.name,
                agent_result.iterations,
                step_result.duration_seconds,
            )
        except TimeoutError:
            step_result.mark_failed(f"Timed out after {step_def.timeout_seconds}s")
            logger.error("Step '{}' timed out", step_def.name)
        except Exception as exc:
            step_result.mark_failed(str(exc))
            logger.error("Step '{}' failed: {}", step_def.name, exc)

    @staticmethod
    def _resolve_template(prompt: str, step_results: dict[str, StepResult]) -> str:
        """Replace {{step_name.output}} placeholders with actual step outputs."""

        def replacer(match: re.Match) -> str:
            step_name = match.group(1)
            result = step_results.get(step_name)
            if result and result.status == StepStatus.COMPLETED:
                return result.output
            return match.group(0)

        return TEMPLATE_PATTERN.sub(replacer, prompt)

    @staticmethod
    def _skip_dependents(
        failed_layer: list[str],
        remaining_layers: list[list[str]],
        result: WorkflowRunResult,
        step_map: dict[str, StepDef],
    ) -> None:
        """Mark steps that depend on failed steps as skipped."""
        failed_set = {
            name for name in failed_layer if result.step_results[name].status == StepStatus.FAILED
        }

        for layer in remaining_layers:
            for step_name in layer:
                step_def = step_map[step_name]
                if any(dep in failed_set for dep in step_def.depends_on):
                    result.step_results[step_name].status = StepStatus.SKIPPED
                    result.step_results[step_name].error = "Skipped due to dependency failure"
                    failed_set.add(step_name)
