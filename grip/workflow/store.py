"""Workflow definition persistence: load/save YAML files from workspace/workflows/.

Each workflow is stored as a JSON file (YAML-like readability with json.dumps indent).
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

    def save(self, workflow: WorkflowDef) -> Path:
        """Save a workflow definition to disk."""
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
        """Load a workflow by name."""
        path = self._dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowDef.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load workflow '{}': {}", name, exc)
            return None

    def list_workflows(self) -> list[str]:
        """Return names of all saved workflows."""
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def delete(self, name: str) -> bool:
        """Delete a workflow by name."""
        path = self._dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
