"""File system tools: read, write, edit, append, list, delete directory.

All path operations go through _resolve_path() which enforces
workspace sandboxing when restrict_to_workspace is enabled.
When restrict_to_workspace is False, the trust system controls
access to directories outside the workspace.
Delete operations move files to a date-stamped trash directory
instead of permanent removal.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext


def _get_trash_dir(workspace: Path) -> Path:
    """Return today's trash directory: workspace/.trash/YYYY-MM-DD/"""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return workspace / ".trash" / today


def _structured_download_path(filename: str, workspace: Path) -> Path:
    """Return a date-stamped download path: workspace/Downloads/YYYY-MM-DD/filename."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return workspace / "Downloads" / today / filename


def _resolve_path(raw_path: str, ctx: ToolContext) -> Path:
    """Resolve a user-provided path against the workspace.

    Relative paths are resolved against workspace_path. When
    restrict_to_workspace is True, any path that escapes the
    workspace is rejected.

    Raises ValueError on path traversal violations.
    """
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = ctx.workspace_path / p
    resolved = p.resolve()

    if ctx.restrict_to_workspace:
        workspace_resolved = ctx.workspace_path.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            raise ValueError(
                f"Path '{raw_path}' resolves outside workspace. Workspace: {workspace_resolved}"
            )

    return resolved


async def _ensure_trusted(resolved: Path, ctx: ToolContext) -> str | None:
    """Check if a resolved path is in a trusted directory.

    Returns None if access is allowed, or an error string if denied.
    Workspace paths are always allowed. For paths outside the workspace,
    the TrustManager is consulted and may prompt the user interactively.
    """
    trust_mgr = ctx.extra.get("trust_manager")
    if trust_mgr is None:
        return None

    from grip.trust import TrustManager

    if not isinstance(trust_mgr, TrustManager):
        return None

    granted = await trust_mgr.check_and_prompt(resolved, ctx.workspace_path)
    if granted:
        return None

    target = TrustManager.find_trust_target(resolved)
    return (
        f"Error: Access denied — '{target}' is not a trusted directory. "
        f"The agent can only access the workspace ({ctx.workspace_path}) "
        f"and explicitly trusted directories. "
        f"Use /trust {target} to grant access."
    )


class ReadFileTool(Tool):
    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Supports optional line offset and limit for large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read (relative to workspace or absolute).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based). Omit to read from the beginning.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Omit to read the entire file.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if not resolved.is_file():
            return f"Error: File not found: {resolved}"

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return f"Error: Permission denied reading {resolved}"

        lines = content.splitlines(keepends=True)
        offset = params.get("offset")
        limit = params.get("limit")

        if offset is not None:
            start = max(0, offset - 1)
            lines = lines[start:]
        if limit is not None:
            lines = lines[:limit]

        result = "".join(lines)
        if len(result) > 100_000:
            result = result[:100_000] + "\n[truncated at 100,000 characters]"
        return result


class WriteFileTool(Tool):
    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create or overwrite a file with the given content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write (relative to workspace or absolute).",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would write {len(params['content'])} characters to {resolved}"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            tmp = resolved.with_suffix(resolved.suffix + ".tmp")
            tmp.write_text(params["content"], encoding="utf-8")
            tmp.rename(resolved)
            return f"Wrote {len(params['content'])} characters to {resolved}"
        except PermissionError:
            return f"Error: Permission denied writing {resolved}"


class EditFileTool(Tool):
    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Find and replace a specific text string within a file. The old_text must appear exactly once."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must occur exactly once in the file).",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if not resolved.is_file():
            return f"Error: File not found: {resolved}"

        try:
            content = resolved.read_text(encoding="utf-8")
        except PermissionError:
            return f"Error: Permission denied reading {resolved}"

        old_text = params["old_text"]
        new_text = params["new_text"]
        count = content.count(old_text)

        if count == 0:
            return f"Error: old_text not found in {resolved}"
        if count > 1:
            return f"Error: old_text appears {count} times in {resolved}. It must be unique."

        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would edit {resolved}: replace 1 occurrence of old_text"

        updated = content.replace(old_text, new_text, 1)
        try:
            tmp = resolved.with_suffix(resolved.suffix + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            tmp.rename(resolved)
            return f"Edited {resolved}: replaced 1 occurrence"
        except PermissionError:
            return f"Error: Permission denied writing {resolved}"


class AppendFileTool(Tool):
    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "append_file"

    @property
    def description(self) -> str:
        return "Append content to the end of a file. Creates the file if it does not exist."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to append to.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to append.",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would append {len(params['content'])} characters to {resolved}"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with resolved.open("a", encoding="utf-8") as f:
                f.write(params["content"])
            return f"Appended {len(params['content'])} characters to {resolved}"
        except PermissionError:
            return f"Error: Permission denied appending to {resolved}"


class ListDirTool(Tool):
    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory with file sizes and types."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (relative to workspace or absolute).",
                },
            },
            "required": ["path"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if not resolved.is_dir():
            return f"Error: Not a directory: {resolved}"

        try:
            entries: list[str] = []
            for item in sorted(resolved.iterdir()):
                try:
                    stat = item.stat()
                    if item.is_dir():
                        entries.append(f"  {item.name}/")
                    else:
                        size = _human_size(stat.st_size)
                        entries.append(f"  {item.name}  ({size})")
                except OSError:
                    entries.append(f"  {item.name}  [access error]")

            if not entries:
                return f"Directory is empty: {resolved}"
            return f"Contents of {resolved}:\n" + "\n".join(entries)
        except PermissionError:
            return f"Error: Permission denied listing {resolved}"


def _human_size(nbytes: int) -> str:
    """Convert byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


class DeleteFileTool(Tool):
    """Safe file/directory deletion that moves to a date-stamped trash folder."""

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return (
            "Delete a file or directory by moving it to the trash folder "
            "(.trash/YYYY-MM-DD/ in the workspace). Not a permanent delete."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to delete (moved to trash).",
                },
            },
            "required": ["path"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            resolved = _resolve_path(params["path"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(resolved, ctx)
        if trust_err:
            return trust_err

        if not resolved.exists():
            return f"Error: Path not found: {resolved}"

        if ctx.extra.get("dry_run"):
            item_type = "directory" if resolved.is_dir() else "file"
            return f"[DRY RUN] Would move {item_type} to trash: {resolved}"

        # Block deletion of critical workspace directories
        protected = {"memory", "sessions", "skills", "cron", "state", "logs", ".trash"}
        workspace_resolved = ctx.workspace_path.resolve()
        if resolved.parent == workspace_resolved and resolved.name in protected:
            return f"Error: Cannot delete protected workspace directory: {resolved.name}"

        trash_dir = _get_trash_dir(ctx.workspace_path)
        trash_dir.mkdir(parents=True, exist_ok=True)

        # Append a numeric suffix if a file with the same name is already in trash
        dest = trash_dir / resolved.name
        counter = 1
        while dest.exists():
            stem = resolved.stem
            suffix = resolved.suffix if resolved.is_file() else ""
            dest = trash_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            shutil.move(str(resolved), str(dest))
            item_type = "directory" if dest.is_dir() else "file"
            logger.info("Trashed {} -> {}", resolved, dest)
            return f"Moved {item_type} to trash: {dest}"
        except PermissionError:
            return f"Error: Permission denied deleting {resolved}"
        except OSError as exc:
            return f"Error: Failed to move to trash: {exc}"


class TrashListTool(Tool):
    """List contents of the workspace trash folder."""

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "trash_list"

    @property
    def description(self) -> str:
        return "List files in the trash folder, grouped by deletion date."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        trash_root = ctx.workspace_path / ".trash"
        if not trash_root.exists():
            return "Trash is empty."

        entries: list[str] = []
        for date_dir in sorted(trash_root.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            items = sorted(date_dir.iterdir())
            if not items:
                continue
            entries.append(f"\n{date_dir.name}:")
            for item in items:
                if item.is_dir():
                    entries.append(f"  {item.name}/")
                else:
                    size = _human_size(item.stat().st_size)
                    entries.append(f"  {item.name}  ({size})")

        if not entries:
            return "Trash is empty."
        return "Trash contents:" + "\n".join(entries)


class TrashRestoreTool(Tool):
    """Restore a file from trash back to its original location."""

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "trash_restore"

    @property
    def description(self) -> str:
        return "Restore a file from trash to a target path in the workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file in trash to restore.",
                },
                "date": {
                    "type": "string",
                    "description": "Trash date folder (YYYY-MM-DD). Defaults to most recent.",
                },
                "restore_to": {
                    "type": "string",
                    "description": "Target path to restore to (relative to workspace or absolute).",
                },
            },
            "required": ["filename", "restore_to"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        trash_root = ctx.workspace_path / ".trash"
        filename = params["filename"]
        date = params.get("date")

        # Find the file in trash
        source: Path | None = None
        if date:
            candidate = trash_root / date / filename
            if candidate.exists():
                source = candidate
        else:
            # Search most recent date folders first
            if trash_root.exists():
                for date_dir in sorted(trash_root.iterdir(), reverse=True):
                    candidate = date_dir / filename
                    if candidate.exists():
                        source = candidate
                        break

        if source is None:
            return f"Error: '{filename}' not found in trash."

        try:
            dest = _resolve_path(params["restore_to"], ctx)
        except ValueError as exc:
            return f"Error: {exc}"

        trust_err = await _ensure_trusted(dest, ctx)
        if trust_err:
            return trust_err

        if dest.exists():
            return f"Error: Target path already exists: {dest}"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            logger.info("Restored from trash: {} -> {}", source, dest)
            return f"Restored to: {dest}"
        except OSError as exc:
            return f"Error: Failed to restore: {exc}"


class SaveFileTool(Tool):
    """Save content to the structured Downloads directory (Downloads/YYYY-MM-DD/filename)."""

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def name(self) -> str:
        return "save_file"

    @property
    def description(self) -> str:
        return "Save content to Downloads/YYYY-MM-DD/filename. Use for generated outputs, reports, and exports."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to save (e.g. 'report.csv', 'output.json').",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["filename", "content"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        filename = params["filename"]
        content = params["content"]

        dest = _structured_download_path(filename, ctx.workspace_path)

        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would save {len(content)} characters to {dest}"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Append a numeric suffix if a file with the same name already exists
            final = dest
            counter = 1
            while final.exists():
                stem = dest.stem
                suffix = dest.suffix
                final = dest.parent / f"{stem}_{counter}{suffix}"
                counter += 1

            tmp = final.with_suffix(final.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.rename(final)
            return f"Saved {len(content)} characters to {final}"
        except PermissionError:
            return f"Error: Permission denied writing {dest}"


def create_filesystem_tools() -> list[Tool]:
    """Factory that returns all filesystem tool instances."""
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        AppendFileTool(),
        ListDirTool(),
        DeleteFileTool(),
        TrashListTool(),
        TrashRestoreTool(),
        SaveFileTool(),
    ]
