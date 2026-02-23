"""Workflow definition persistence: load/save JSON files from workspace/workflows/.

Each workflow is stored as a JSON file with readable indentation.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from grip.workflow.models import WorkflowDef


class WorkflowStore:
    """Manages workflow definition files on disk."""

    def __init__(self, workflows_dir: Path) -> None:
        self._dir = workflows_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject names containing path traversal characters or empty strings."""
        if not name or not name.strip():
            raise ValueError("Workflow name cannot be empty")
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"Invalid workflow name: {name!r}")

    def save(self, workflow: WorkflowDef) -> Path:
        """Save a workflow definition to disk."""
        self._validate_name(workflow.name)
        path = self._dir / f"{workflow.name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(workflow.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.rename(path)
        logger.debug("Saved workflow: {}", workflow.name)
        return path

    def load(self, name: str) -> WorkflowDef | None:
        """Load a workflow by name. Returns None if not found or invalid."""
        try:
            self._validate_name(name)
        except ValueError:
            return None
        path = self._dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowDef.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to load workflow '{}': {}", name, exc)
            return None

    def list_workflows(self) -> list[str]:
        """Return names of all saved workflows."""
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def delete(self, name: str) -> bool:
        """Delete a workflow by name."""
        try:
            self._validate_name(name)
        except ValueError:
            return False
        path = self._dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
