"""Shell command execution tool with safety guards.

Runs commands via asyncio subprocess with configurable timeout, working
directory enforcement, and a deny-list of dangerous command patterns.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext

_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Destructive file operations
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*$",  # rm -rf /
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\*",  # rm -rf /*
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+~",  # rm -rf ~ (any variant)
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+\$HOME",  # rm -rf $HOME
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/home\b",  # rm -rf /home
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/etc\b",  # rm -rf /etc
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/var\b",  # rm -rf /var
        r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/usr\b",  # rm -rf /usr
        # Disk/device destruction
        r"mkfs\b",
        r"dd\s+if=",
        r">\s*/dev/sd[a-z]",
        r">\s*/dev/nvme",
        r">\s*/dev/disk",
        # Fork bombs and system control
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\binit\s+[06]\b",
        r"\bsystemctl\s+(poweroff|reboot|halt)\b",
        # Permission escalation on system dirs
        r"chmod\s+-R\s+777\s+/\s*$",
        r"chmod\s+-R\s+000\s+/",
        r"chmod\s+000\s+/",
        r"chown\s+-R\s+.*\s+/\s*$",
        r"chattr\s+\+i\s+/",
        # Piped execution of remote code
        r"curl\b.*\|\s*(ba)?sh\b",
        r"wget\b.*\|\s*(ba)?sh\b",
        r"curl\b.*\|\s*python",
        r"wget\b.*\|\s*python",
        r"curl\b.*\|\s*perl",
        # Credential/key extraction
        r"cat\s+.*\.ssh/id_",
        r"cat\s+.*\.env\b",
        r"cat\s+.*/\.aws/credentials",
        r"cat\s+.*/\.netrc",
        # History theft
        r"cat\s+.*\.(bash_|zsh_)?history",
        # Network exfiltration of sensitive files
        r"curl\b.*-[a-z]*d\s*@.*\.(env|pem|key)\b",
        r"scp\s+.*\.(env|pem|key)\s",
    )
)

_OUTPUT_LIMIT = 50_000


def _is_dangerous(command: str) -> str | None:
    """Check if a command matches any deny pattern.

    Returns the matched pattern description, or None if safe.
    """
    for pattern in _DENY_PATTERNS:
        if pattern.search(command):
            return f"Command blocked: matches dangerous pattern '{pattern.pattern}'"
    return None


class ShellTool(Tool):
    @property
    def category(self) -> str:
        return "shell"

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout/stderr."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Defaults to the configured shell_timeout.",
                },
            },
            "required": ["command"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        command = params["command"]
        timeout = params.get("timeout") or ctx.shell_timeout

        danger = _is_dangerous(command)
        if danger:
            logger.warning("Blocked dangerous command: {}", command)
            return f"Error: {danger}"

        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would execute: {command} (timeout={timeout}s)"

        cwd = str(ctx.workspace_path)
        logger.info("Executing shell: {} (timeout={}s, cwd={})", command, timeout, cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: Command timed out after {timeout}s: {command}"

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            parts: list[str] = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            if proc.returncode != 0:
                parts.append(f"[exit code: {proc.returncode}]")

            output = "\n".join(parts) if parts else "(no output)"

            if len(output) > _OUTPUT_LIMIT:
                half = _OUTPUT_LIMIT // 2
                output = (
                    output[:half]
                    + f"\n\n[... truncated {len(output) - _OUTPUT_LIMIT} chars ...]\n\n"
                    + output[-half:]
                )

            return output

        except FileNotFoundError:
            return f"Error: Shell not found. Cannot execute: {command}"
        except OSError as exc:
            return f"Error: OS error executing command: {exc}"


def create_shell_tools() -> list[Tool]:
    return [ShellTool()]
