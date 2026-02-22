"""Context builder: assembles the system prompt from workspace files.

Reads identity files (AGENT.md, IDENTITY.md, SOUL.md, USER.md, SHIELD.md),
appends a compact skill listing and metadata, and returns a single
system message ready for the LLM.

Tool definitions are already sent via the API's ``tools`` parameter, so
the system prompt only carries identity, skills overview, shield policy,
tone hints, and runtime metadata — keeping token usage minimal.
"""

from __future__ import annotations

import platform
import re
from datetime import UTC, datetime

from loguru import logger

from grip import __version__
from grip.providers.types import LLMMessage
from grip.workspace.manager import WorkspaceManager

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

    is_caps = (
        sum(1 for c in user_message if c.isupper()) > len(user_message) * 0.6
        and len(user_message) > 10
    )
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
      1. Agent identity (AGENT.md + IDENTITY.md + SOUL.md + USER.md)
      2. Shield policy (SHIELD.md — runtime threat feed evaluation rules)
      3. Compact skill listing (name + one-line description only)
      4. Tone adaptation (based on user message sentiment)
      5. Runtime metadata (datetime, platform, version)

    Tool definitions are NOT included here — they travel via the API's
    ``tools`` parameter and are therefore excluded to save tokens.
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
        user_message: str = "",
        session_key: str = "",
    ) -> LLMMessage:
        """Assemble the full system prompt and return it as an LLMMessage."""
        parts: list[str] = []

        identity = self._build_identity_section()
        if identity:
            parts.append(identity)

        skills_listing = self._build_skills_listing()
        if skills_listing:
            parts.append(skills_listing)

        tone_hint = _detect_tone_hint(user_message)
        if tone_hint:
            parts.append(tone_hint)

        parts.append(self._build_metadata_section(session_key=session_key))

        system_prompt = "\n\n---\n\n".join(parts)
        logger.debug("System prompt built: {} chars", len(system_prompt))
        return LLMMessage(role="system", content=system_prompt)

    def _build_identity_section(self) -> str:
        """Concatenate identity + shield files into a single block.

        Includes AGENT.md, IDENTITY.md, SOUL.md, USER.md, and SHIELD.md.
        Caches the result since these files rarely change mid-session.
        """
        if self._cached_identity is not None:
            return self._cached_identity

        identity_files = self._workspace.read_identity_files()
        sections: list[str] = []

        for filename in ("AGENT.md", "IDENTITY.md", "SOUL.md", "USER.md", "SHIELD.md"):
            content = identity_files.get(filename)
            if content and content.strip():
                sections.append(content.strip())

        self._cached_identity = "\n\n".join(sections)
        return self._cached_identity

    def _build_skills_listing(self) -> str:
        """Build a compact name+description listing of available skills.

        Full skill content is NOT injected — the LLM can load a skill's
        instructions on demand via the read_file tool when needed.
        """
        from grip.skills.loader import SkillsLoader

        try:
            loader = SkillsLoader(self._workspace.root)
            skills = loader.scan()
        except Exception as exc:
            logger.debug("Failed to scan skills for system prompt: {}", exc)
            return ""

        if not skills:
            return ""

        lines = ["## Available Skills\n"]
        for s in skills:
            desc = f": {s.description}" if s.description else ""
            lines.append(f"- **{s.name}**{desc}")
        lines.append("\nUse the read_file tool to load a skill's full instructions when needed.")
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
