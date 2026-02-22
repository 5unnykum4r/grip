"""Document generation tool — template-based document creation.

Built-in templates stored as Python string constants for: report, readme,
changelog, meeting_notes, and custom freeform. Variable substitution uses
``{key}`` placeholders. Output formats: markdown and html.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grip.tools.base import Tool, ToolContext

_TEMPLATES: dict[str, str] = {
    "report": (
        "# {title}\n\n"
        "**Author:** {author}\n"
        "**Date:** {date}\n\n"
        "## Executive Summary\n\n{summary}\n\n"
        "## Details\n\n{details}\n\n"
        "## Conclusions\n\n{conclusions}\n"
    ),
    "readme": (
        "# {project_name}\n\n"
        "{description}\n\n"
        "## Installation\n\n```bash\n{install_command}\n```\n\n"
        "## Usage\n\n{usage}\n\n"
        "## License\n\n{license}\n"
    ),
    "changelog": (
        "# Changelog\n\n"
        "## [{version}] - {date}\n\n"
        "### Added\n{added}\n\n"
        "### Changed\n{changed}\n\n"
        "### Fixed\n{fixed}\n"
    ),
    "meeting_notes": (
        "# Meeting Notes: {title}\n\n"
        "**Date:** {date}\n"
        "**Attendees:** {attendees}\n\n"
        "## Agenda\n\n{agenda}\n\n"
        "## Discussion\n\n{discussion}\n\n"
        "## Action Items\n\n{action_items}\n"
    ),
    "custom": "{content}",
}


def _substitute_variables(template: str, variables: dict[str, str]) -> str:
    """Replace {key} placeholders with provided values, leaving unmatched ones as-is."""
    variables.setdefault("date", datetime.now(UTC).strftime("%Y-%m-%d"))
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def _markdown_to_html(markdown: str) -> str:
    """Minimal markdown-to-HTML conversion for document output."""
    lines = markdown.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_list = False

    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">' if lang else "<pre><code>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list and not line.strip():
                html_lines.append("</ul>")
                in_list = False
            escaped = html.escape(line)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
            escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
            html_lines.append(f"<p>{escaped}</p>" if line.strip() else "")

    if in_list:
        html_lines.append("</ul>")
    if in_code_block:
        html_lines.append("</code></pre>")

    body = "\n".join(html_lines)
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        "<style>body{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;}"
        "pre{background:#f4f4f4;padding:16px;border-radius:4px;overflow-x:auto;}"
        "code{background:#f4f4f4;padding:2px 6px;border-radius:3px;}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>"
    )


class DocumentGenTool(Tool):
    """Template-based document generation with variable substitution."""

    @property
    def name(self) -> str:
        return "document_gen"

    @property
    def description(self) -> str:
        return (
            "Generate documents from templates (report, readme, changelog, meeting_notes, custom)."
        )

    @property
    def category(self) -> str:
        return "general"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "enum": ["report", "readme", "changelog", "meeting_notes", "custom"],
                    "description": "Built-in template name.",
                },
                "variables": {
                    "type": "object",
                    "description": "Key-value pairs for template variable substitution.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["markdown", "html"],
                    "description": "Output format. Default: markdown.",
                    "default": "markdown",
                },
                "output_file": {
                    "type": "string",
                    "description": "Optional file path to save the document.",
                },
            },
            "required": ["template", "variables"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        template_name = params.get("template", "custom")
        variables = params.get("variables", {})
        output_format = params.get("output_format", "markdown")
        output_file = params.get("output_file", "")

        template = _TEMPLATES.get(template_name)
        if template is None:
            return f"Error: unknown template '{template_name}'. Available: {', '.join(_TEMPLATES.keys())}"

        markdown_output = _substitute_variables(template, variables)

        if output_format == "html":
            final_output = _markdown_to_html(markdown_output)
        else:
            final_output = markdown_output

        if output_file:
            out_path = Path(output_file)
            if not out_path.is_absolute():
                out_path = ctx.workspace_path / out_path
            if ctx.restrict_to_workspace:
                try:
                    out_path.resolve().relative_to(ctx.workspace_path.resolve())
                except ValueError:
                    return "Error: output path is outside the workspace sandbox."
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(final_output, encoding="utf-8")
            return f"Document saved to {out_path}\n\n{final_output[:2000]}"

        return final_output


def create_document_gen_tools() -> list[Tool]:
    """Factory function returning document generation tool instances."""
    return [DocumentGenTool()]
