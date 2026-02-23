"""Shell command execution tool with multi-layer safety guards.

Runs commands via asyncio subprocess with configurable timeout, working
directory enforcement, and a multi-layer deny system for dangerous commands.

Safety layers:
  1. Blocked base commands (mkfs, shutdown, reboot, etc.)
  2. Parsed rm detection with normalized short/long flags and dangerous targets
  3. Interpreter -c escape detection (python, bash, perl with inline code)
  4. Regex fallback for patterns that are hard to parse structurally
     (fork bombs, pipe-to-shell, credential access, device writes)
"""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext

# ---------------------------------------------------------------------------
# Layer 1: Commands that are always dangerous regardless of arguments
# ---------------------------------------------------------------------------
_BLOCKED_COMMANDS: frozenset[str] = frozenset({
    "mkfs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4",
    "mkfs.xfs", "mkfs.btrfs", "mkfs.vfat", "mkfs.ntfs",
    "shutdown", "reboot", "halt", "poweroff",
})

_BLOCKED_SYSTEMCTL_ACTIONS: frozenset[str] = frozenset({
    "poweroff", "reboot", "halt",
})

# ---------------------------------------------------------------------------
# Layer 2: rm flag normalization and dangerous target detection
# ---------------------------------------------------------------------------
_RM_LONG_FLAG_MAP: dict[str, str] = {
    "--recursive": "r",
    "--force": "f",
    "--interactive": "i",
    "--dir": "d",
    "--verbose": "v",
    "--no-preserve-root": "!",
}

_DANGEROUS_RM_TARGETS: tuple[str, ...] = (
    "/", "/*",
    "~", "$HOME",
    "/home", "/etc", "/var", "/usr", "/bin", "/sbin",
    "/lib", "/boot", "/root", "/opt", "/srv",
)

# ---------------------------------------------------------------------------
# Layer 3: Interpreters that can execute arbitrary code via -c
# ---------------------------------------------------------------------------
_INTERPRETER_COMMANDS: frozenset[str] = frozenset({
    "python", "python3", "python3.10", "python3.11", "python3.12", "python3.13",
    "bash", "sh", "zsh", "dash", "ksh", "fish",
    "perl", "ruby", "node", "lua",
})

# ---------------------------------------------------------------------------
# Layer 4: Regex fallback for patterns hard to parse structurally
# ---------------------------------------------------------------------------
_REGEX_DENY: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Fork bombs
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",
        # dd to disk devices
        r"\bdd\s+if=",
        # Redirect to block devices
        r">\s*/dev/sd[a-z]",
        r">\s*/dev/nvme",
        r">\s*/dev/disk",
        # Permission escalation on root
        r"\bchmod\s+.*\s+/\s*$",
        r"\bchown\s+.*\s+/\s*$",
        r"\bchattr\s+\+i\s+/",
        # Piped execution of remote code
        r"\bcurl\b.*\|\s*(ba)?sh\b",
        r"\bwget\b.*\|\s*(ba)?sh\b",
        r"\bcurl\b.*\|\s*python",
        r"\bwget\b.*\|\s*python",
        r"\bcurl\b.*\|\s*perl",
        # Credential file access
        r"\bcat\s+.*\.ssh/id_",
        r"\bcat\s+.*\.env\b",
        r"\bcat\s+.*/\.aws/credentials",
        r"\bcat\s+.*/\.netrc",
        # History theft
        r"\bcat\s+.*\.(bash_|zsh_)?history",
        # Network exfiltration of sensitive files
        r"\bcurl\b.*-[a-z]*d\s*@.*\.(env|pem|key)\b",
        r"\bscp\s+.*\.(env|pem|key)\s",
    )
)

_OUTPUT_LIMIT = 50_000

# Maximum recursion depth for interpreter escape checking
_MAX_CHECK_DEPTH = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_shell_commands(command: str) -> list[str]:
    """Split a shell command string on ; && || operators into subcommands.

    Respects single and double quoting so that separators inside strings
    are not treated as command boundaries.
    """
    parts: list[str] = []
    current: list[str] = []
    i = 0
    in_single = False
    in_double = False

    while i < len(command):
        ch = command[i]

        if ch == "\\" and i + 1 < len(command) and not in_single:
            current.append(ch)
            current.append(command[i + 1])
            i += 2
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if command[i : i + 2] == "&&":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 2
                continue
            if command[i : i + 2] == "||":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 2
                continue
            if ch == ";":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _tokenize(command: str) -> list[str]:
    """Tokenize a command with shlex, falling back to split on parse error."""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _strip_sudo(tokens: list[str]) -> list[str]:
    """Strip leading 'sudo' (with optional flags like -u user) from tokens."""
    if not tokens or tokens[0] != "sudo":
        return tokens
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 1
        if i < len(tokens):
            i += 1
    return tokens[i:] if i < len(tokens) else tokens


def _extract_rm_flags(tokens: list[str]) -> set[str]:
    """Extract normalized single-char flags from rm arguments."""
    flags: set[str] = set()
    for token in tokens[1:]:
        if token == "--":
            break
        if token.startswith("--"):
            mapped = _RM_LONG_FLAG_MAP.get(token)
            if mapped:
                flags.add(mapped)
        elif token.startswith("-") and len(token) > 1 and not token[1:].isdigit():
            for ch in token[1:]:
                flags.add(ch)
    return flags


def _extract_rm_targets(tokens: list[str]) -> list[str]:
    """Extract non-flag arguments (file/dir targets) from rm tokens."""
    targets: list[str] = []
    past_flags = False
    for token in tokens[1:]:
        if token == "--":
            past_flags = True
            continue
        if past_flags or not token.startswith("-"):
            targets.append(token)
    return targets


def _check_rm(tokens: list[str]) -> str | None:
    """Check if an rm command is dangerous based on parsed flags and targets."""
    flags = _extract_rm_flags(tokens)
    targets = _extract_rm_targets(tokens)

    if "!" in flags and "r" in flags:
        return "rm with --no-preserve-root and recursive flag"

    has_recursive = "r" in flags
    has_force = "f" in flags

    for target in targets:
        normalized = target.rstrip("/") or "/"
        if has_recursive and normalized == "/":
            return "rm -r on root filesystem"
        if has_recursive and has_force:
            for dangerous in _DANGEROUS_RM_TARGETS:
                if normalized == dangerous or normalized == dangerous.rstrip("/"):
                    return f"rm -rf on critical path: {target}"
    return None


_INTERPRETER_CODE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"rm\s.*-.*r.*-.*f.*\s+/",
        r"rm\s+-rf\s",
        r"rm\s+--recursive",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\bmkfs\b",
        r"\.ssh/id_",
        r"\.env\b",
        r"/\.aws/credentials",
        r"\.(bash_|zsh_)?history",
    )
)


def _check_interpreter(tokens: list[str], base_cmd: str, depth: int) -> str | None:
    """Check if an interpreter -c command executes dangerous inline code.

    Uses two strategies:
      1. Recursive shell-level check (catches bash -c "rm -rf /")
      2. Regex scan of raw code argument (catches python3 -c "os.system('rm -rf /')")
    """
    code_arg = None
    for i, token in enumerate(tokens):
        if token == "-c" and i + 1 < len(tokens):
            code_arg = tokens[i + 1]
            break
        if token.startswith("-c") and len(token) > 2:
            code_arg = token[2:]
            break

    if base_cmd == "eval" and len(tokens) > 1 and code_arg is None:
        code_arg = " ".join(tokens[1:])

    if code_arg is None:
        return None

    # Strategy 1: treat the code as shell and check recursively
    danger = _is_dangerous(code_arg, _depth=depth + 1)
    if danger:
        return f"Interpreter escape via {base_cmd} -c: {danger}"

    # Strategy 2: regex scan for dangerous patterns embedded in code strings
    for pattern in _INTERPRETER_CODE_PATTERNS:
        if pattern.search(code_arg):
            return f"Interpreter escape via {base_cmd} -c: code contains '{pattern.pattern}'"

    # Strategy 3: also check the regex deny layer on the code argument
    for pattern in _REGEX_DENY:
        if pattern.search(code_arg):
            return f"Interpreter escape via {base_cmd} -c: {pattern.pattern}"

    return None


def _is_dangerous(command: str, *, _depth: int = 0) -> str | None:
    """Multi-layer check for dangerous shell commands.

    Returns a description of why the command is blocked, or None if safe.
    """
    if _depth >= _MAX_CHECK_DEPTH:
        return None

    for subcmd in _split_shell_commands(command):
        tokens = _tokenize(subcmd)
        if not tokens:
            continue

        tokens = _strip_sudo(tokens)
        if not tokens:
            continue

        base_cmd = tokens[0].rsplit("/", maxsplit=1)[-1]

        # Layer 1: Unconditionally blocked commands
        if base_cmd in _BLOCKED_COMMANDS:
            return f"Blocked command: {base_cmd}"

        if base_cmd == "systemctl" and len(tokens) > 1 and tokens[1] in _BLOCKED_SYSTEMCTL_ACTIONS:
                return f"systemctl {tokens[1]} is blocked"

        if base_cmd == "init" and len(tokens) > 1 and tokens[1] in ("0", "6"):
            return f"init {tokens[1]} (system halt/reboot)"

        # Layer 2: rm with parsed flags
        if base_cmd == "rm":
            result = _check_rm(tokens)
            if result:
                return result

        # Layer 3: Interpreter -c escape
        if base_cmd in _INTERPRETER_COMMANDS or base_cmd == "eval":
            result = _check_interpreter(tokens, base_cmd, _depth)
            if result:
                return result

    # Layer 4: Regex fallback on the full command string
    for pattern in _REGEX_DENY:
        if pattern.search(command):
            return f"Blocked: matches pattern '{pattern.pattern}'"

    return None


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

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
