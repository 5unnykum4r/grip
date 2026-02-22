"""Code analysis tool — AST-based static analysis for Python files.

Uses Python's built-in ``ast`` module (zero external dependencies) to compute:
- **complexity**: Cyclomatic complexity per function (if/elif/for/while/except/and/or)
- **dependencies**: Import graph split into stdlib, third-party, and local imports
- **structure**: Function/class counts, size distribution, max nesting depth
"""

from __future__ import annotations

import ast
import contextlib
import sys
from pathlib import Path
from typing import Any

from grip.tools.base import Tool, ToolContext

_MAX_FILES = 50

_STDLIB_TOP_LEVEL: frozenset[str] | None = None


def _get_stdlib_modules() -> frozenset[str]:
    """Return the set of top-level stdlib module names for the running Python."""
    global _STDLIB_TOP_LEVEL  # noqa: PLW0603
    if _STDLIB_TOP_LEVEL is not None:
        return _STDLIB_TOP_LEVEL
    try:
        _STDLIB_TOP_LEVEL = frozenset(sys.stdlib_module_names)
    except AttributeError:
        _STDLIB_TOP_LEVEL = frozenset(
            {
                "os",
                "sys",
                "re",
                "json",
                "math",
                "pathlib",
                "typing",
                "collections",
                "functools",
                "itertools",
                "io",
                "datetime",
                "time",
                "logging",
                "unittest",
                "ast",
                "abc",
                "dataclasses",
                "contextlib",
                "subprocess",
                "threading",
                "asyncio",
                "http",
                "urllib",
                "socket",
                "hashlib",
                "secrets",
                "uuid",
                "csv",
                "xml",
                "email",
                "html",
                "textwrap",
                "shutil",
                "tempfile",
            }
        )
    return _STDLIB_TOP_LEVEL


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Count decision points in an AST node (cyclomatic complexity).

    Counts: if, elif (via chained if), for, while, except, and, or, assert,
    with (context managers can raise), ternary (IfExp).
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child, (ast.If, ast.IfExp, ast.For, ast.While, ast.AsyncFor, ast.ExceptHandler)
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _max_nesting_depth(node: ast.AST, current: int = 0) -> int:
    """Compute maximum nesting depth of control structures."""
    max_depth = current
    nesting_nodes = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_nodes):
            child_depth = _max_nesting_depth(child, current + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _max_nesting_depth(child, current)
            max_depth = max(max_depth, child_depth)
    return max_depth


def _analyze_complexity(tree: ast.AST, source_lines: int) -> dict[str, Any]:
    """Analyze cyclomatic complexity for every function/method in the module."""
    functions: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _cyclomatic_complexity(node)
            end_line = getattr(node, "end_lineno", node.lineno)
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": cc,
                    "lines": end_line - node.lineno + 1,
                }
            )

    functions.sort(key=lambda f: f["complexity"], reverse=True)
    total_cc = sum(f["complexity"] for f in functions)
    avg_cc = total_cc / len(functions) if functions else 0

    return {
        "file_lines": source_lines,
        "function_count": len(functions),
        "average_complexity": round(avg_cc, 2),
        "total_complexity": total_cc,
        "functions": functions[:20],
    }


def _analyze_dependencies(tree: ast.AST) -> dict[str, list[str]]:
    """Categorize imports into stdlib, third_party, and local."""
    stdlib = _get_stdlib_modules()
    result: dict[str, set[str]] = {"stdlib": set(), "third_party": set(), "local": set()}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in stdlib:
                    result["stdlib"].add(top)
                else:
                    result["third_party"].add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                module_name = node.module or "(relative)"
                result["local"].add(module_name)
            elif node.module:
                top = node.module.split(".")[0]
                if top in stdlib:
                    result["stdlib"].add(top)
                else:
                    result["third_party"].add(top)

    return {k: sorted(v) for k, v in result.items()}


def _analyze_structure(tree: ast.AST, source_lines: int) -> dict[str, Any]:
    """Count classes, functions, and compute size/nesting metrics."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    func_sizes = []
    for f in functions:
        end_line = getattr(f, "end_lineno", f.lineno)
        func_sizes.append(end_line - f.lineno + 1)

    max_nesting = _max_nesting_depth(tree)

    return {
        "file_lines": source_lines,
        "class_count": len(classes),
        "function_count": len(functions),
        "max_nesting_depth": max_nesting,
        "avg_function_size": round(sum(func_sizes) / len(func_sizes), 1) if func_sizes else 0,
        "largest_function": max(func_sizes) if func_sizes else 0,
        "smallest_function": min(func_sizes) if func_sizes else 0,
    }


def _format_result(analysis_type: str, file_path: str, data: dict[str, Any]) -> str:
    """Format analysis results as readable text."""
    lines = [f"## {analysis_type.title()} Analysis: {file_path}\n"]

    if analysis_type == "complexity":
        lines.append(f"File lines: {data['file_lines']}")
        lines.append(f"Functions: {data['function_count']}")
        lines.append(f"Average complexity: {data['average_complexity']}")
        lines.append(f"Total complexity: {data['total_complexity']}")
        lines.append("")
        if data["functions"]:
            lines.append("| Function | Line | Complexity | Lines |")
            lines.append("|----------|------|------------|-------|")
            for f in data["functions"]:
                lines.append(f"| {f['name']} | {f['line']} | {f['complexity']} | {f['lines']} |")
    elif analysis_type == "dependencies":
        for category in ("stdlib", "third_party", "local"):
            deps = data.get(category, [])
            lines.append(f"**{category}** ({len(deps)}): {', '.join(deps) if deps else 'none'}")
    elif analysis_type == "structure":
        for key, value in data.items():
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {value}")

    return "\n".join(lines)


class CodeAnalysisTool(Tool):
    """AST-based static analysis for Python source files."""

    @property
    def name(self) -> str:
        return "code_analysis"

    @property
    def description(self) -> str:
        return "AST-based Python analysis: cyclomatic complexity, import graph, and code structure."

    @property
    def category(self) -> str:
        return "general"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to analyze. Relative paths resolve from workspace.",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["complexity", "dependencies", "structure"],
                    "description": "Type of analysis to perform.",
                },
            },
            "required": ["path", "analysis_type"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        raw_path = params.get("path", "")
        analysis_type = params.get("analysis_type", "complexity")

        if analysis_type not in ("complexity", "dependencies", "structure"):
            return "Error: analysis_type must be one of: complexity, dependencies, structure"

        target = Path(raw_path)
        if not target.is_absolute():
            target = ctx.workspace_path / target

        if ctx.restrict_to_workspace:
            try:
                target.resolve().relative_to(ctx.workspace_path.resolve())
            except ValueError:
                return f"Error: path '{raw_path}' is outside the workspace sandbox."

        if not target.exists():
            return f"Error: path '{raw_path}' does not exist."

        files: list[Path] = []
        if target.is_file():
            if target.suffix != ".py":
                return "Error: code_analysis only supports Python (.py) files."
            files = [target]
        else:
            files = sorted(target.rglob("*.py"))[:_MAX_FILES]
            if not files:
                return f"No Python files found in '{raw_path}'."

        results: list[str] = []
        for file_path in files:
            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(file_path))
                source_lines = len(source.splitlines())
            except SyntaxError as exc:
                results.append(f"## {file_path}: SyntaxError at line {exc.lineno}")
                continue
            except Exception as exc:
                results.append(f"## {file_path}: Error reading file: {exc}")
                continue

            display_path = str(file_path)
            with contextlib.suppress(ValueError):
                display_path = str(file_path.relative_to(ctx.workspace_path))

            if analysis_type == "complexity":
                data = _analyze_complexity(tree, source_lines)
            elif analysis_type == "dependencies":
                data = _analyze_dependencies(tree)
            else:
                data = _analyze_structure(tree, source_lines)

            results.append(_format_result(analysis_type, display_path, data))

        if len(files) > 1:
            header = f"Analyzed {len(files)} Python files ({analysis_type})\n\n"
            return header + "\n\n---\n\n".join(results)

        return "\n".join(results)


def create_code_analysis_tools() -> list[Tool]:
    """Factory function returning code analysis tool instances."""
    return [CodeAnalysisTool()]
