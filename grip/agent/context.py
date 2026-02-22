"""Context builder: assembles the system prompt from workspace files.

Reads identity files (AGENT.md, IDENTITY.md, SOUL.md, USER.md, MEMORY.md),
appends available tool descriptions and metadata, and returns a single
system message ready for the LLM.

Tool and skill summaries come from TOOLS.md (auto-generated at startup by
grip.tools.docs). If TOOLS.md is missing, falls back to inline generation.
"""

from __future__ import annotations

import platform
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from grip import __version__
from grip.providers.types import LLMMessage
from grip.workspace.manager import WorkspaceManager

if TYPE_CHECKING:
    from grip.tools.base import ToolRegistry


# Patterns used by _detect_tone_hint() to classify the user's emotional state.
_ERROR_PATTERNS = re.compile(
    r"(traceback|error|exception|failed|crash|bug|broken|not working|won't work)",
    re.IGNORECASE,
)
_FRUSTRATION_PATTERNS = re.compile(
    r"(wtf|damn|hell|ugh|fuck|shit|crap|stupid|hate|awful|terrible|why won't)",
    re.IGNORECASE,
)
_BRAINSTORM_PATTERNS = re.compile(
    r"(idea|brainstorm|what if|how could|design|architect|plan|explore|suggest|creative)",
    re.IGNORECASE,
)


def _detect_tone_hint(user_message: str) -> str:
    """Return a short tone instruction based on the user's message content.

    Returns empty string when no special tone is warranted.
    """
    if not user_message:
        return ""

    is_caps = sum(1 for c in user_message if c.isupper()) > len(user_message) * 0.6 and len(user_message) > 10
    frustrated = bool(_FRUSTRATION_PATTERNS.search(user_message)) or is_caps
    has_error = bool(_ERROR_PATTERNS.search(user_message))
    brainstorming = bool(_BRAINSTORM_PATTERNS.search(user_message))

    if frustrated and has_error:
        return (
            "## Tone Adaptation\n\n"
            "The user seems frustrated with an error. "
            "Be calm, precise, and surgical. Lead with the fix, not explanations. "
            "Show empathy briefly, then focus on solving the problem step by step."
        )
    if frustrated:
        return (
            "## Tone Adaptation\n\n"
            "The user seems stressed. Be patient and supportive. "
            "Break things into small, clear steps. Avoid jargon. "
            "Confirm understanding before proceeding."
        )
    if has_error:
        return (
            "## Tone Adaptation\n\n"
            "The user is dealing with an error. "
            "Be concise and action-oriented. Diagnose first, then provide a clear fix."
        )
    if brainstorming:
        return (
            "## Tone Adaptation\n\n"
            "The user is brainstorming. Be expansive and creative. "
            "Suggest multiple approaches, trade-offs, and alternatives. "
            "Encourage exploration."
        )
    return ""


class ContextBuilder:
    """Builds a system prompt from workspace files and runtime metadata.

    The system prompt follows this structure:
      1. Agent identity (AGENT.md + IDENTITY.md + SOUL.md)
      2. User context (USER.md)
      3. Long-term memory (MEMORY.md)
      4. Tools & skills manifest (from TOOLS.md)
      5. Always-loaded skill content
      6. Runtime metadata (datetime, platform)
    """

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace
        self._cached_identity: str | None = None

    def invalidate_cache(self) -> None:
        """Force re-read of identity files on next build."""
        self._cached_identity = None

    def build_system_message(
        self,
        *,
        tool_definitions: list[dict[str, Any]] | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_names: list[str] | None = None,
        user_message: str = "",
        session_key: str = "",
    ) -> LLMMessage:
        """Assemble the full system prompt and return it as an LLMMessage."""
        parts: list[str] = []

        identity = self._build_identity_section()
        if identity:
            parts.append(identity)

        # Prefer pre-generated TOOLS.md over inline tool/skill summaries
        tools_md = self._workspace.read_file("TOOLS.md")
        if tools_md and tools_md.strip():
            parts.append(tools_md.strip())
        elif tool_definitions:
            # Fallback: inline generation when TOOLS.md hasn't been written yet
            parts.append(self._summarize_tools_inline(tool_definitions, tool_registry))
            if skill_names:
                skills_list = ", ".join(skill_names)
                parts.append(
                    f"## Available Skills\n\n"
                    f"You have access to these skills: {skills_list}\n"
                    f"Use the read_file tool to load a skill's full instructions when needed."
                )

        always_loaded = self._workspace.read_builtin_skills()
        if always_loaded:
            parts.append(always_loaded)

        # Dynamic persona: adapt tone based on user's current message
        tone_hint = _detect_tone_hint(user_message)
        if tone_hint:
            parts.append(tone_hint)

        parts.append(self._build_metadata_section(session_key=session_key))

        system_prompt = "\n\n---\n\n".join(parts)
        logger.debug("System prompt built: {} chars", len(system_prompt))
        return LLMMessage(role="system", content=system_prompt)

    def _build_identity_section(self) -> str:
        """Concatenate AGENT.md, IDENTITY.md, and SOUL.md into a single identity block.

        Caches the result since these files rarely change mid-session.
        """
        if self._cached_identity is not None:
            return self._cached_identity

        identity_files = self._workspace.read_identity_files()
        sections: list[str] = []

        for filename in ("AGENT.md", "IDENTITY.md", "SOUL.md", "USER.md"):
            content = identity_files.get(filename)
            if content and content.strip():
                sections.append(content.strip())

        self._cached_identity = "\n\n".join(sections)
        return self._cached_identity

    @staticmethod
    def _summarize_tools_inline(
        tool_definitions: list[dict[str, Any]],
        tool_registry: ToolRegistry | None = None,
    ) -> str:
        """Fallback inline tool summary when TOOLS.md is not available."""
        lines: list[str] = ["## Available Tools"]
        for tool_def in tool_definitions:
            fn = tool_def.get("function", tool_def)
            lines.append(_format_tool_line(fn))
        return "\n".join(lines)

    @staticmethod
    def _build_metadata_section(session_key: str = "") -> str:
        now = datetime.now(UTC)
        lines = [
            "## Runtime Info\n",
            f"- Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Platform: {platform.system()} {platform.release()}",
            f"- Python: {platform.python_version()}",
            f"- grip version: {__version__}",
        ]
        if session_key:
            lines.append(f"- Session key: {session_key}")
        return "\n".join(lines)


def _format_tool_line(fn: dict[str, Any]) -> str:
    """Format a single tool definition into a markdown line."""
    name = fn.get("name", "unknown")
    desc = fn.get("description", "No description")
    params = fn.get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})

    param_parts: list[str] = []
    for pname, pschema in properties.items():
        ptype = pschema.get("type", "any")
        marker = " (required)" if pname in required else ""
        param_parts.append(f"{pname}: {ptype}{marker}")

    params_str = ", ".join(param_parts) if param_parts else "none"
    return f"- **{name}**({params_str}): {desc}"
