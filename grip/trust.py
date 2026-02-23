"""Directory trust manager for filesystem access control.

When restrict_to_workspace is False, grip uses a trust-based model:
- The workspace directory is always trusted.
- Other directories must be explicitly trusted by the user.
- Trusted directories include all their subdirectories.
- Trust decisions are persisted to workspace/state/trusted_dirs.json.

In CLI mode, the user is prompted interactively (Trust / Deny).
In gateway mode, untrusted paths return an error and the user
can grant trust via the /trust command.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger

TrustPrompt = Callable[[Path], Awaitable[bool]]


class TrustManager:
    """Manages per-directory trust for filesystem tool access."""

    def __init__(self, state_dir: Path) -> None:
        self._state_file = state_dir / "trusted_dirs.json"
        self._trusted: set[str] = set()
        self._denied_this_session: set[str] = set()
        self._lock = asyncio.Lock()
        self._prompt: TrustPrompt | None = None
        self._load()

    def _load(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            self._trusted = set(data.get("directories", []))
            logger.debug("Loaded {} trusted directories", len(self._trusted))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load trusted_dirs.json: {}", exc)
            self._trusted = set()

    def _save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"directories": sorted(self._trusted)}
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(self._state_file)

    def set_prompt(self, callback: TrustPrompt) -> None:
        """Set the async callback used to prompt the user for trust decisions."""
        self._prompt = callback

    @property
    def trusted_directories(self) -> list[str]:
        """Return sorted list of trusted directory paths."""
        return sorted(self._trusted)

    def is_trusted(self, path: Path, workspace: Path) -> bool:
        """Check if a resolved path falls within the workspace or a trusted directory."""
        resolved = path.resolve()
        ws = workspace.resolve()

        # Workspace is always trusted
        if resolved == ws or _is_subpath(resolved, ws):
            return True

        # Check each trusted directory
        for td in self._trusted:
            td_path = Path(td)
            if resolved == td_path or _is_subpath(resolved, td_path):
                return True

        return False

    @staticmethod
    def find_trust_target(path: Path) -> Path:
        """Determine the top-level directory to trust for a given path.

        For paths under the user's home directory, returns ~/first_child
        (e.g., ~/Downloads for ~/Downloads/project/file.txt).
        For paths outside home, returns the first directory component
        (e.g., /tmp for /tmp/work/file.txt).
        """
        resolved = path.resolve()
        home = Path.home().resolve()

        try:
            relative = resolved.relative_to(home)
            if relative.parts:
                return home / relative.parts[0]
            return resolved
        except ValueError:
            # Outside home — trust the first directory after root
            if len(resolved.parts) > 1:
                return Path(resolved.root) / resolved.parts[1]
            return resolved

    def trust(self, directory: Path) -> None:
        """Permanently trust a directory and all its subdirectories."""
        resolved_str = str(directory.resolve())
        self._trusted.add(resolved_str)
        self._denied_this_session.discard(resolved_str)
        self._save()
        logger.info("Trusted directory: {}", resolved_str)

    def revoke(self, directory: Path) -> bool:
        """Remove a directory from the trusted list. Returns True if it was trusted."""
        resolved_str = str(directory.resolve())
        if resolved_str in self._trusted:
            self._trusted.discard(resolved_str)
            self._save()
            logger.info("Revoked trust for: {}", resolved_str)
            return True
        return False

    async def check_and_prompt(self, path: Path, workspace: Path) -> bool:
        """Check trust and prompt the user if needed.

        Returns True if access is granted, False if denied.
        Uses an asyncio lock to prevent multiple concurrent prompts
        for the same directory when tools run in parallel.
        """
        if self.is_trusted(path, workspace):
            return True

        target = self.find_trust_target(path)
        target_str = str(target)

        # Already denied this session — don't re-prompt
        if target_str in self._denied_this_session:
            return False

        async with self._lock:
            # Re-check after acquiring lock (a parallel tool may have trusted it)
            if self.is_trusted(path, workspace):
                return True
            if target_str in self._denied_this_session:
                return False

            # No prompt callback (gateway/API mode) — deny silently
            if self._prompt is None:
                return False

            granted = await self._prompt(target)
            if granted:
                self.trust(target)
                return True
            else:
                self._denied_this_session.add(target_str)
                return False


def _is_subpath(child: Path, parent: Path) -> bool:
    """Check if child is a subdirectory/file within parent (Python 3.9+ safe)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
