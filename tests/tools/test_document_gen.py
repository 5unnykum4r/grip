"""Tests for the document_gen tool."""

from __future__ import annotations

import pytest

from grip.tools.base import ToolContext
from grip.tools.document_gen import (
    DocumentGenTool,
    _markdown_to_html,
    _substitute_variables,
    _TEMPLATES,
    create_document_gen_tools,
)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


class TestSubstituteVariables:
    def test_replaces_known_variables(self):
        result = _substitute_variables("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_leaves_unknown_variables(self):
        result = _substitute_variables("Hello {name}!", {})
        assert "{name}" in result

    def test_auto_fills_date(self):
        result = _substitute_variables("Date: {date}", {})
        assert "{date}" not in result
        assert "20" in result


class TestMarkdownToHtml:
    def test_converts_headings(self):
        html = _markdown_to_html("# Title\n## Subtitle")
        assert "<h1>Title</h1>" in html
        assert "<h2>Subtitle</h2>" in html

    def test_wraps_in_html_document(self):
        html = _markdown_to_html("Hello")
        assert "<!DOCTYPE html>" in html
        assert "<body>" in html

    def test_handles_bold(self):
        html = _markdown_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in html

    def test_handles_code_blocks(self):
        html = _markdown_to_html("```python\nprint('hi')\n```")
        assert "<pre><code" in html
        assert "print" in html


class TestTemplates:
    def test_all_templates_are_strings(self):
        for name, tmpl in _TEMPLATES.items():
            assert isinstance(tmpl, str), f"Template '{name}' is not a string"

    def test_report_template_has_sections(self):
        tmpl = _TEMPLATES["report"]
        assert "{title}" in tmpl
        assert "{summary}" in tmpl
        assert "{details}" in tmpl

    def test_readme_template_has_sections(self):
        tmpl = _TEMPLATES["readme"]
        assert "{project_name}" in tmpl
        assert "{install_command}" in tmpl


class TestDocumentGenTool:
    def test_factory_returns_tool(self):
        tools = create_document_gen_tools()
        assert len(tools) == 1
        assert tools[0].name == "document_gen"

    @pytest.mark.asyncio
    async def test_report_template_produces_markdown(self, ctx):
        tool = DocumentGenTool()
        result = await tool.execute({
            "template": "report",
            "variables": {
                "title": "Q4 Report",
                "author": "Test",
                "summary": "Good quarter.",
                "details": "Revenue up.",
                "conclusions": "Keep going.",
            },
        }, ctx)
        assert "Q4 Report" in result
        assert "Good quarter." in result

    @pytest.mark.asyncio
    async def test_html_output_format(self, ctx):
        tool = DocumentGenTool()
        result = await tool.execute({
            "template": "custom",
            "variables": {"content": "# Hello World"},
            "output_format": "html",
        }, ctx)
        assert "<!DOCTYPE html>" in result

    @pytest.mark.asyncio
    async def test_saves_to_file(self, ctx):
        tool = DocumentGenTool()
        result = await tool.execute({
            "template": "custom",
            "variables": {"content": "Test content"},
            "output_file": "docs/test.md",
        }, ctx)
        assert (ctx.workspace_path / "docs" / "test.md").exists()

    @pytest.mark.asyncio
    async def test_unknown_template_returns_error(self, ctx):
        tool = DocumentGenTool()
        result = await tool.execute({
            "template": "nonexistent",
            "variables": {},
        }, ctx)
        assert "Error" in result
