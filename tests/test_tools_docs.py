"""Tests for TOOLS.md generation and skill frontmatter parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grip.skills.loader import Skill, SkillsLoader
from grip.tools.base import Tool, ToolContext, ToolRegistry
from grip.tools.docs import generate_tools_md

# ---------------------------------------------------------------------------
# Helpers: minimal tool subclasses for testing
# ---------------------------------------------------------------------------


class _FinanceTool(Tool):
    @property
    def name(self) -> str:
        return "stock_quote"

    @property
    def description(self) -> str:
        return "Get real-time stock quote"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"symbols": {"type": "string"}},
            "required": ["symbols"],
        }

    @property
    def category(self) -> str:
        return "finance"

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        return "ok"


class _FileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a file from the workspace"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    @property
    def category(self) -> str:
        return "filesystem"

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        return "ok"


def _make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    category: str = "utility",
    always_loaded: bool = False,
) -> Skill:
    return Skill(
        name=name,
        description=description,
        content="skill content here",
        source_path=Path(f"/fake/{name}/SKILL.md"),
        always_loaded=always_loaded,
        category=category,
    )


# ---------------------------------------------------------------------------
# generate_tools_md tests
# ---------------------------------------------------------------------------


class TestGenerateToolsMd:
    def test_returns_valid_markdown_with_expected_sections(self):
        registry = _make_registry(_FinanceTool(), _FileTool())
        skills = [_make_skill()]
        result = generate_tools_md(registry, skills)

        assert "# grip — Available Tools & Skills" in result
        assert "## Tool Usage Guidelines" in result
        assert "## Built-in Tools" in result
        assert "## Skills" in result

    def test_tool_categories_grouped(self):
        registry = _make_registry(_FinanceTool(), _FileTool())
        result = generate_tools_md(registry, [])

        assert "### Finance" in result
        assert "### Filesystem" in result
        assert "| `stock_quote`" in result
        assert "| `read_file`" in result

    def test_skill_categories_grouped(self):
        skills = [
            _make_skill("cron", "Schedule tasks", "automation", always_loaded=True),
            _make_skill("debug", "Debug grip", "debugging"),
        ]
        result = generate_tools_md(_make_registry(), skills)

        assert "### Automation" in result
        assert "### Debugging" in result
        assert "| cron |" in result
        assert "| Yes |" in result
        assert "| debug |" in result
        assert "| No |" in result

    def test_mcp_servers_section(self):
        class _MCPStub:
            url = "http://localhost:3000"
            command = ""
            args: list[str] = []

        servers = {"my-server": _MCPStub()}
        result = generate_tools_md(_make_registry(), [], mcp_servers=servers)

        assert "## MCP Servers" in result
        assert "| my-server |" in result
        assert "http://localhost:3000" in result

    def test_mcp_section_omitted_when_empty(self):
        result = generate_tools_md(_make_registry(), [], mcp_servers={})
        assert "## MCP Servers" not in result

    def test_skills_section_omitted_when_empty(self):
        result = generate_tools_md(_make_registry(), [])
        assert "## Skills" not in result

    def test_tool_parameters_listed(self):
        registry = _make_registry(_FinanceTool())
        result = generate_tools_md(registry, [])
        assert "symbols (required)" in result


# ---------------------------------------------------------------------------
# Frontmatter parsing tests
# ---------------------------------------------------------------------------


class TestFrontmatterParsing:
    def test_yaml_frontmatter_parsed(self, tmp_path: Path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "title: My Skill\n"
            "description: Does amazing things\n"
            "category: automation\n"
            "always_loaded: true\n"
            "---\n"
            "# My Skill\n\n"
            "Content goes here.\n"
        )
        skill = SkillsLoader._parse_skill_file(skill_dir / "SKILL.md")

        assert skill is not None
        assert skill.name == "My Skill"
        assert skill.description == "Does amazing things"
        assert skill.category == "automation"
        assert skill.always_loaded is True
        assert "Content goes here." in skill.content

    def test_legacy_h1_blockquote_fallback(self, tmp_path: Path):
        skill_dir = tmp_path / "old-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# Old Skill\n\n> This skill does old stuff.\n\n## Instructions\nDo the thing.\n"
        )
        skill = SkillsLoader._parse_skill_file(skill_dir / "SKILL.md")

        assert skill is not None
        assert skill.name == "Old Skill"
        assert skill.description == "This skill does old stuff."
        assert skill.category == "general"
        assert skill.always_loaded is False

    def test_legacy_always_loaded_comment(self, tmp_path: Path):
        skill_dir = tmp_path / "loaded-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# Loaded\n\n> Always active skill.\n<!-- always_loaded -->\n"
        )
        skill = SkillsLoader._parse_skill_file(skill_dir / "SKILL.md")

        assert skill is not None
        assert skill.always_loaded is True

    def test_frontmatter_name_field_compat(self, tmp_path: Path):
        """The `name` field in frontmatter should work as fallback for `title`."""
        skill_dir = tmp_path / "compat"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: compat-tool\ndescription: backward compatible\n---\nBody.\n"
        )
        skill = SkillsLoader._parse_skill_file(skill_dir / "SKILL.md")

        assert skill is not None
        assert skill.name == "compat-tool"
        assert skill.category == "general"

    def test_frontmatter_title_takes_precedence_over_name(self, tmp_path: Path):
        skill_dir = tmp_path / "both"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ntitle: Proper Title\nname: slug-name\ndescription: has both\n---\nBody.\n"
        )
        skill = SkillsLoader._parse_skill_file(skill_dir / "SKILL.md")

        assert skill is not None
        assert skill.name == "Proper Title"

    def test_empty_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "empty.md"
        p.write_text("")
        assert SkillsLoader._parse_skill_file(p) is None

    def test_category_field_on_skill_dataclass(self):
        skill = Skill(
            name="test",
            description="desc",
            content="body",
            source_path=Path("/fake"),
            category="debugging",
        )
        assert skill.category == "debugging"

    def test_category_defaults_to_general(self):
        skill = Skill(
            name="test",
            description="desc",
            content="body",
            source_path=Path("/fake"),
        )
        assert skill.category == "general"


# ---------------------------------------------------------------------------
# Integration: verify builtin skills parse with frontmatter
# ---------------------------------------------------------------------------


class TestBuiltinSkillsParse:
    def test_all_builtin_skills_have_frontmatter(self):
        """All 15 builtin skills should parse with YAML frontmatter and have a category."""
        builtin_dir = Path(__file__).parent.parent / "grip" / "skills" / "builtin"
        if not builtin_dir.exists():
            pytest.skip("builtin skills directory not found")

        skill_files = sorted(builtin_dir.glob("*/SKILL.md"))
        assert len(skill_files) >= 15, f"Expected 15+ skills, found {len(skill_files)}"

        for path in skill_files:
            skill = SkillsLoader._parse_skill_file(path)
            assert skill is not None, f"Failed to parse {path}"
            assert skill.name, f"Missing name in {path}"
            assert skill.description, f"Missing description in {path}"
            assert skill.category != "general", (
                f"{path.parent.name} should have an explicit category, got 'general'"
            )
