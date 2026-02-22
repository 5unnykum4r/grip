"""Tests for the code_analysis tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from grip.tools.base import ToolContext
from grip.tools.code_analysis import (
    CodeAnalysisTool,
    _analyze_complexity,
    _analyze_dependencies,
    _analyze_structure,
    _cyclomatic_complexity,
    _max_nesting_depth,
    create_code_analysis_tools,
)

import ast


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


SIMPLE_FUNCTION = """\
def greet(name):
    return f"Hello, {name}"
"""

COMPLEX_FUNCTION = """\
def process(data, mode):
    if mode == "fast":
        for item in data:
            if item > 0:
                yield item
            elif item == 0:
                continue
            else:
                raise ValueError("negative")
    elif mode == "slow":
        while data:
            try:
                val = data.pop()
            except IndexError:
                break
            if val > 0 and val < 100:
                yield val
"""

IMPORT_SAMPLE = """\
import os
import json
import httpx
from pathlib import Path
from . import utils
from .models import User
"""


class TestCyclomaticComplexity:
    def test_simple_function_has_low_complexity(self):
        tree = ast.parse(SIMPLE_FUNCTION)
        func = tree.body[0]
        cc = _cyclomatic_complexity(func)
        assert cc == 1

    def test_complex_function_has_high_complexity(self):
        tree = ast.parse(COMPLEX_FUNCTION)
        func = tree.body[0]
        cc = _cyclomatic_complexity(func)
        assert cc > 5


class TestNestingDepth:
    def test_simple_function_low_nesting(self):
        tree = ast.parse(SIMPLE_FUNCTION)
        depth = _max_nesting_depth(tree)
        assert depth <= 1

    def test_complex_function_deeper_nesting(self):
        tree = ast.parse(COMPLEX_FUNCTION)
        depth = _max_nesting_depth(tree)
        assert depth >= 2


class TestAnalyzeComplexity:
    def test_returns_function_list(self):
        tree = ast.parse(SIMPLE_FUNCTION + "\ndef other(): pass\n")
        result = _analyze_complexity(tree, 5)
        assert result["function_count"] == 2
        assert len(result["functions"]) == 2

    def test_sorted_by_complexity_descending(self):
        source = SIMPLE_FUNCTION + "\n" + COMPLEX_FUNCTION
        tree = ast.parse(source)
        result = _analyze_complexity(tree, len(source.splitlines()))
        complexities = [f["complexity"] for f in result["functions"]]
        assert complexities == sorted(complexities, reverse=True)


class TestAnalyzeDependencies:
    def test_classifies_stdlib(self):
        tree = ast.parse(IMPORT_SAMPLE)
        deps = _analyze_dependencies(tree)
        assert "os" in deps["stdlib"]
        assert "json" in deps["stdlib"]
        assert "pathlib" in deps["stdlib"]

    def test_classifies_third_party(self):
        tree = ast.parse(IMPORT_SAMPLE)
        deps = _analyze_dependencies(tree)
        assert "httpx" in deps["third_party"]

    def test_classifies_local(self):
        tree = ast.parse(IMPORT_SAMPLE)
        deps = _analyze_dependencies(tree)
        assert len(deps["local"]) >= 1


class TestAnalyzeStructure:
    def test_counts_functions(self):
        source = "def a(): pass\ndef b(): pass\nclass C:\n    def d(self): pass\n"
        tree = ast.parse(source)
        result = _analyze_structure(tree, 4)
        assert result["function_count"] == 3
        assert result["class_count"] == 1


class TestCodeAnalysisTool:
    def test_factory_returns_tool(self):
        tools = create_code_analysis_tools()
        assert len(tools) == 1
        assert tools[0].name == "code_analysis"

    @pytest.mark.asyncio
    async def test_analyzes_python_file(self, ctx):
        py_file = ctx.workspace_path / "sample.py"
        py_file.write_text(SIMPLE_FUNCTION, encoding="utf-8")

        tool = CodeAnalysisTool()
        result = await tool.execute(
            {"path": "sample.py", "analysis_type": "complexity"}, ctx
        )
        assert "greet" in result

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_path(self, ctx):
        tool = CodeAnalysisTool()
        result = await tool.execute(
            {"path": "nonexistent.py", "analysis_type": "complexity"}, ctx
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_non_python_file(self, ctx):
        txt_file = ctx.workspace_path / "readme.txt"
        txt_file.write_text("hello", encoding="utf-8")

        tool = CodeAnalysisTool()
        result = await tool.execute(
            {"path": "readme.txt", "analysis_type": "complexity"}, ctx
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_directory_scan(self, ctx):
        pkg = ctx.workspace_path / "pkg"
        pkg.mkdir()
        (pkg / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
        (pkg / "b.py").write_text("def bar(): pass\n", encoding="utf-8")

        tool = CodeAnalysisTool()
        result = await tool.execute(
            {"path": "pkg", "analysis_type": "structure"}, ctx
        )
        assert "Analyzed 2 Python files" in result
