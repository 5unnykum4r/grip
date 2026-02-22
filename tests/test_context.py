"""Tests for ContextBuilder — system prompt assembly and token optimization.

Verifies that the system prompt:
  - Includes identity files
  - Includes a compact skill listing (names + descriptions only)
  - Does NOT include TOOLS.md content (tool definitions travel via API)
  - Does NOT include full always-loaded skill content
  - Includes runtime metadata and tone hints
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grip.agent.context import ContextBuilder, _detect_tone_hint
from grip.workspace.manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    """Create a minimal workspace with identity files and a skill."""
    ws = WorkspaceManager(tmp_path)
    ws.initialize()

    # Write a skill with always_loaded flag
    skill_dir = tmp_path / "skills" / "test-cron"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "title: test-cron\n"
        "description: Schedule tasks and reminders\n"
        "category: automation\n"
        "always_loaded: true\n"
        "---\n"
        "## Full Instructions\n\n"
        "This is a very long set of instructions that should NOT be in the prompt.\n"
        "It contains multiple paragraphs of detailed workflow steps.\n"
        "Line 3 of instructions.\n"
        "Line 4 of instructions.\n"
        "Line 5 of instructions.\n",
    )

    # Write a second non-always-loaded skill
    skill_dir2 = tmp_path / "skills" / "test-debug"
    skill_dir2.mkdir(parents=True)
    (skill_dir2 / "SKILL.md").write_text(
        "---\n"
        "title: test-debug\n"
        "description: Debug grip issues\n"
        "category: debugging\n"
        "---\n"
        "Debug instructions body.\n",
    )

    return ws


@pytest.fixture
def builder(workspace: WorkspaceManager) -> ContextBuilder:
    return ContextBuilder(workspace)


# ---------------------------------------------------------------------------
# TOOLS.md exclusion
# ---------------------------------------------------------------------------


class TestToolsMdExcluded:
    """TOOLS.md content must NOT appear in the system prompt."""

    def test_no_tools_md_content_in_prompt(
        self, workspace: WorkspaceManager, builder: ContextBuilder
    ):
        (workspace.root / "TOOLS.md").write_text(
            "# grip — Available Tools & Skills\n\n"
            "## Tool Usage Guidelines\n\n"
            "IMPORTANT: Always prefer specialized tools.\n\n"
            "## Built-in Tools\n\n"
            "| Tool | Parameters | Description |\n",
        )
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "Tool Usage Guidelines" not in prompt
        assert "Built-in Tools" not in prompt
        assert "grip — Available Tools" not in prompt

    def test_no_inline_tool_summary_in_prompt(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "## Available Tools" not in prompt


# ---------------------------------------------------------------------------
# Skills: compact listing only (no full content)
# ---------------------------------------------------------------------------


class TestSkillsCompactListing:
    """System prompt should list skill names, not full content."""

    def test_skill_names_listed(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "test-cron" in prompt
        assert "test-debug" in prompt

    def test_skill_descriptions_listed(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "Schedule tasks and reminders" in prompt
        assert "Debug grip issues" in prompt

    def test_full_skill_content_excluded(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "Full Instructions" not in prompt
        assert "very long set of instructions" not in prompt
        assert "Debug instructions body" not in prompt

    def test_read_file_hint_present(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "read_file" in prompt
        assert "skill" in prompt.lower()

    def test_available_skills_section_present(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello")
        prompt = msg.content

        assert "## Available Skills" in prompt


# ---------------------------------------------------------------------------
# Identity files
# ---------------------------------------------------------------------------


class TestIdentitySection:
    def test_identity_files_included(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hi")
        prompt = msg.content

        assert "Agent Guidelines" in prompt
        assert "Identity" in prompt

    def test_identity_caching(self, builder: ContextBuilder):
        msg1 = builder.build_system_message(user_message="hi")
        msg2 = builder.build_system_message(user_message="hello again")

        assert msg1.content.startswith(msg2.content[:50])

    def test_invalidate_cache(self, builder: ContextBuilder):
        builder.build_system_message(user_message="hi")
        assert builder._cached_identity is not None

        builder.invalidate_cache()
        assert builder._cached_identity is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_section_present(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hi", session_key="cli:test")
        prompt = msg.content

        assert "## Runtime Info" in prompt
        assert "grip version" in prompt
        assert "cli:test" in prompt

    def test_metadata_includes_platform(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hi")
        prompt = msg.content

        assert "Platform:" in prompt
        assert "Python:" in prompt


# ---------------------------------------------------------------------------
# Tone hints
# ---------------------------------------------------------------------------


class TestToneHints:
    def test_no_tone_for_simple_message(self):
        assert _detect_tone_hint("What is 2+2?") == ""

    def test_error_tone_detected(self):
        result = _detect_tone_hint("I got a traceback error")
        assert "error" in result.lower()

    def test_frustration_tone_detected(self):
        result = _detect_tone_hint("wtf is going on with this")
        assert "stressed" in result.lower() or "frustrated" in result.lower()

    def test_brainstorm_tone_detected(self):
        result = _detect_tone_hint("I have an idea for a new design")
        assert "brainstorm" in result.lower() or "creative" in result.lower()

    def test_tone_included_in_prompt(self, builder: ContextBuilder):
        msg = builder.build_system_message(
            user_message="This traceback error is driving me crazy wtf"
        )
        prompt = msg.content

        assert "## Tone Adaptation" in prompt


# ---------------------------------------------------------------------------
# Token savings: prompt size regression guard
# ---------------------------------------------------------------------------


class TestTokenSavings:
    """Guard against system prompt growing beyond the optimized size."""

    def test_prompt_size_under_limit(self, builder: ContextBuilder):
        msg = builder.build_system_message(user_message="hello", session_key="cli:test")
        prompt_len = len(msg.content)

        # Identity files (~2KB) + skill listing (~3KB for ~30 skills) + metadata (~200 bytes)
        # Total should be well under 8KB (was ~43KB before optimization)
        assert prompt_len < 8000, (
            f"System prompt is {prompt_len} chars — expected under 8000 after optimization"
        )

    def test_build_system_message_signature_simplified(self, builder: ContextBuilder):
        """Ensure old parameters (tool_definitions, tool_registry, skill_names) are gone."""
        import inspect

        sig = inspect.signature(builder.build_system_message)
        param_names = set(sig.parameters.keys())

        assert "tool_definitions" not in param_names
        assert "tool_registry" not in param_names
        assert "skill_names" not in param_names
        assert "user_message" in param_names
        assert "session_key" in param_names
