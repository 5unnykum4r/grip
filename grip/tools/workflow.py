"""Workflow tool — CRUD and execution for DAG-based multi-step workflows.

Allows agents to create, list, inspect, edit, and delete workflow
definitions stored as JSON in the workspace. Workflows define a DAG
of steps (each step runs an agent with a specific profile) and can
be executed via the workflow engine.

Actions: create, list, show, edit, delete.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext
from grip.workflow.models import StepDef, WorkflowDef
from grip.workflow.store import WorkflowStore

_MAX_STEPS = 50


class WorkflowTool(Tool):
    """Manage multi-step workflow definitions from the agent loop."""

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def description(self) -> str:
        return (
            "Create, list, inspect, edit, and delete DAG-based multi-step workflows. "
            "Each workflow is a sequence of agent steps that can depend on each other "
            "and reference prior step outputs via {{step_name.output}} templates."
        )

    @property
    def category(self) -> str:
        return "orchestration"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "show", "edit", "delete"],
                    "description": "Action to perform.",
                },
                "workflow_name": {
                    "type": "string",
                    "description": (
                        "Name of the workflow (for create/show/edit/delete). "
                        "Must be alphanumeric with underscores/hyphens only."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Workflow description (for create/edit).",
                },
                "steps": {
                    "type": "array",
                    "description": (
                        "Step definitions (for create/edit). Each step is an object with: "
                        "name (required), prompt (required), profile (default: 'default'), "
                        "depends_on (list of step names), timeout_seconds (default: 300)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "prompt": {"type": "string"},
                            "profile": {"type": "string"},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "timeout_seconds": {"type": "integer"},
                        },
                        "required": ["name", "prompt"],
                    },
                },
            },
            "required": ["action"],
        }

    def _get_store(self, ctx: ToolContext) -> WorkflowStore:
        return WorkflowStore(ctx.workspace_path / "workflows")

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        action = params.get("action", "")
        store = self._get_store(ctx)

        if action == "create":
            return self._create(params, store)
        elif action == "list":
            return self._list(store)
        elif action == "show":
            return self._show(params, store)
        elif action == "edit":
            return self._edit(params, store)
        elif action == "delete":
            return self._delete(params, store)
        else:
            return f"Error: unknown action '{action}'. Use: create, list, show, edit, delete."

    def _parse_steps(self, raw_steps: list[dict[str, Any]]) -> list[StepDef]:
        """Convert raw step dicts from tool params into StepDef instances."""
        steps: list[StepDef] = []
        for s in raw_steps:
            steps.append(
                StepDef(
                    name=s["name"],
                    prompt=s["prompt"],
                    profile=s.get("profile", "default"),
                    depends_on=s.get("depends_on", []),
                    timeout_seconds=int(s.get("timeout_seconds", 300)),
                )
            )
        return steps

    def _create(self, params: dict[str, Any], store: WorkflowStore) -> str:
        wf_name = params.get("workflow_name", "").strip()
        if not wf_name:
            return "Error: workflow_name is required for create action."

        raw_steps = params.get("steps")
        if not raw_steps or not isinstance(raw_steps, list):
            return "Error: steps array is required for create action (at least one step)."

        if len(raw_steps) > _MAX_STEPS:
            return f"Error: workflow exceeds maximum of {_MAX_STEPS} steps."

        existing = store.load(wf_name)
        if existing:
            return (
                f"Error: workflow '{wf_name}' already exists. "
                "Use action 'edit' to update it, or 'delete' first."
            )

        try:
            steps = self._parse_steps(raw_steps)
        except (KeyError, TypeError, ValueError) as exc:
            return f"Error: invalid step definition: {exc}"

        wf = WorkflowDef(
            name=wf_name,
            description=params.get("description", ""),
            steps=steps,
        )

        errors = wf.validate()
        if errors:
            return "Error: workflow validation failed:\n" + "\n".join(f"  - {e}" for e in errors)

        path = store.save(wf)
        layers = wf.get_execution_order()
        logger.info("Agent created workflow '{}' ({} steps)", wf_name, len(steps))

        return (
            f"Workflow '{wf_name}' created successfully.\n"
            f"  Steps: {len(steps)}\n"
            f"  Execution layers: {len(layers)}\n"
            f"  Saved to: {path}\n"
            f"  Run with: grip workflow run {wf_name}"
        )

    def _list(self, store: WorkflowStore) -> str:
        names = store.list_workflows()
        if not names:
            return "No workflows found."

        lines = ["## Saved Workflows\n"]
        for name in names:
            wf = store.load(name)
            if wf:
                layers = wf.get_execution_order()
                lines.append(
                    f"- **{name}**: {len(wf.steps)} steps, {len(layers)} layers"
                    + (f" — {wf.description}" if wf.description else "")
                )
            else:
                lines.append(f"- **{name}**: (failed to load)")
        return "\n".join(lines)

    def _show(self, params: dict[str, Any], store: WorkflowStore) -> str:
        wf_name = params.get("workflow_name", "").strip()
        if not wf_name:
            return "Error: workflow_name is required for show action."

        wf = store.load(wf_name)
        if not wf:
            return f"Error: workflow '{wf_name}' not found."

        layers = wf.get_execution_order()
        errors = wf.validate()

        lines = [
            f"## Workflow: {wf.name}",
            f"Description: {wf.description or '(none)'}",
            f"Steps: {len(wf.steps)}",
            f"Execution layers: {len(layers)}",
            "",
            "### Steps",
        ]

        for step in wf.steps:
            deps = ", ".join(step.depends_on) if step.depends_on else "(none)"
            lines.append(
                f"- **{step.name}** [profile: {step.profile}, timeout: {step.timeout_seconds}s]\n"
                f"  Dependencies: {deps}\n"
                f"  Prompt: {step.prompt[:200]}{'...' if len(step.prompt) > 200 else ''}"
            )

        lines.append("")
        lines.append("### Execution Order")
        for i, layer in enumerate(layers, 1):
            lines.append(f"  Layer {i}: {', '.join(layer)}")

        if errors:
            lines.append("")
            lines.append("### Validation Errors")
            for err in errors:
                lines.append(f"  - {err}")

        return "\n".join(lines)

    def _edit(self, params: dict[str, Any], store: WorkflowStore) -> str:
        wf_name = params.get("workflow_name", "").strip()
        if not wf_name:
            return "Error: workflow_name is required for edit action."

        existing = store.load(wf_name)
        if not existing:
            return (
                f"Error: workflow '{wf_name}' not found. "
                "Use action 'create' to create a new workflow."
            )

        raw_steps = params.get("steps")
        if not raw_steps or not isinstance(raw_steps, list):
            return "Error: steps array is required for edit action."

        if len(raw_steps) > _MAX_STEPS:
            return f"Error: workflow exceeds maximum of {_MAX_STEPS} steps."

        try:
            steps = self._parse_steps(raw_steps)
        except (KeyError, TypeError, ValueError) as exc:
            return f"Error: invalid step definition: {exc}"

        wf = WorkflowDef(
            name=wf_name,
            description=params.get("description", existing.description),
            steps=steps,
        )

        errors = wf.validate()
        if errors:
            return "Error: workflow validation failed:\n" + "\n".join(f"  - {e}" for e in errors)

        path = store.save(wf)
        layers = wf.get_execution_order()
        logger.info("Agent updated workflow '{}' ({} steps)", wf_name, len(steps))

        return (
            f"Workflow '{wf_name}' updated successfully.\n"
            f"  Steps: {len(steps)}\n"
            f"  Execution layers: {len(layers)}\n"
            f"  Saved to: {path}"
        )

    def _delete(self, params: dict[str, Any], store: WorkflowStore) -> str:
        wf_name = params.get("workflow_name", "").strip()
        if not wf_name:
            return "Error: workflow_name is required for delete action."

        if store.delete(wf_name):
            logger.info("Agent deleted workflow '{}'", wf_name)
            return f"Workflow '{wf_name}' deleted."
        return f"Error: workflow '{wf_name}' not found."


def create_workflow_tools() -> list[Tool]:
    """Factory function returning workflow tool instances."""
    return [WorkflowTool()]
