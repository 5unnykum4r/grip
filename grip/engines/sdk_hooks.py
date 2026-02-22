"""SDK hooks for trust enforcement, security, and memory persistence.

These hooks integrate grip's security model with the Claude Agent SDK:
  - PreToolUse: blocks dangerous shell commands and enforces file access trust
  - PostToolUse: logs tool execution for observability
  - Stop: persists a conversation summary to history after each agent run

The shell deny patterns are imported from grip/tools/shell.py to stay in sync
with the LiteLLM engine's ShellTool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from grip.tools.shell import _DENY_PATTERNS

if TYPE_CHECKING:
    from grip.memory import MemoryManager
    from grip.trust import TrustManager


def build_pre_tool_use_hook(
    workspace_root: Path,
    trust_mgr: TrustManager | None = None,
):
    """Return a PreToolUse hook that enforces trust and blocks dangerous commands.

    When returned dict has ``decision: "block"``, the SDK will skip the tool call
    and surface the ``message`` to the model.  Returning ``None`` means "allow".
    """

    resolved_workspace = workspace_root.resolve()

    def pre_tool_use(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
        # Block dangerous shell commands using the same regex patterns as ShellTool
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            for pattern in _DENY_PATTERNS:
                if pattern.search(command):
                    logger.warning(
                        "SDK hook blocked dangerous command: {} (pattern: {})",
                        command[:100],
                        pattern.pattern,
                    )
                    return {
                        "decision": "block",
                        "message": f"Blocked: matches dangerous pattern '{pattern.pattern}'",
                    }

        # Enforce trust for file operations outside workspace
        if trust_mgr and tool_name in ("Read", "Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path:
                resolved = Path(file_path).expanduser().resolve()
                if not trust_mgr.is_trusted(resolved.parent, resolved_workspace):
                    logger.warning(
                        "SDK hook blocked file access outside trusted dirs: {}",
                        resolved,
                    )
                    return {
                        "decision": "block",
                        "message": (
                            f"Directory not trusted: {resolved.parent}. Use /trust to allow access."
                        ),
                    }

        return None

    return pre_tool_use


def build_post_tool_use_hook():
    """Return a PostToolUse hook that logs tool execution for observability."""

    def post_tool_use(tool_name: str, tool_input: dict[str, Any], tool_output: str) -> None:
        logger.debug("SDK tool executed: {} -> {} chars output", tool_name, len(tool_output))

    return post_tool_use


def build_stop_hook(memory_mgr: MemoryManager | None):
    """Return a Stop hook that persists a conversation summary to history.

    Called by the SDK after the agent run completes. Saves a truncated summary
    so future sessions can recall context from past interactions.
    """

    def stop_hook(conversation_summary: str) -> None:
        if memory_mgr and conversation_summary:
            memory_mgr.append_history(f"[Session summary] {conversation_summary[:500]}")
            logger.debug("Stop hook: saved conversation summary to history")

    return stop_hook
