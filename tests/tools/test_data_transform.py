"""Tests for the data_transform tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from grip.tools.base import ToolContext
from grip.tools.data_transform import (
    DataTransformTool,
    _apply_aggregate,
    _apply_filter,
    _apply_select,
    _apply_sort,
    create_data_transform_tools,
)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


SAMPLE_DATA = [
    {"name": "Alice", "age": "30", "dept": "Engineering"},
    {"name": "Bob", "age": "25", "dept": "Marketing"},
    {"name": "Carol", "age": "35", "dept": "Engineering"},
    {"name": "Dave", "age": "28", "dept": "Marketing"},
]


class TestFilter:
    def test_equals_filter(self):
        result = _apply_filter(SAMPLE_DATA, {"column": "dept", "op": "==", "value": "Engineering"})
        assert len(result) == 2
        assert all(r["dept"] == "Engineering" for r in result)

    def test_greater_than_filter(self):
        result = _apply_filter(SAMPLE_DATA, {"column": "age", "op": ">", "value": 28})
        assert len(result) == 2

    def test_contains_filter(self):
        result = _apply_filter(SAMPLE_DATA, {"column": "name", "op": "contains", "value": "a"})
        assert any(r["name"] == "Carol" for r in result)
        assert any(r["name"] == "Dave" for r in result)


class TestSelect:
    def test_selects_columns(self):
        result = _apply_select(SAMPLE_DATA, ["name", "dept"])
        assert all(set(r.keys()) == {"name", "dept"} for r in result)

    def test_missing_column_skipped(self):
        result = _apply_select(SAMPLE_DATA, ["name", "nonexistent"])
        assert all("nonexistent" not in r for r in result)


class TestSort:
    def test_sort_by_name(self):
        result = _apply_sort(SAMPLE_DATA, {"by": "name"})
        names = [r["name"] for r in result]
        assert names == sorted(names, key=str.lower)

    def test_sort_reverse(self):
        result = _apply_sort(SAMPLE_DATA, {"by": "age", "reverse": True})
        ages = [float(r["age"]) for r in result]
        assert ages == sorted(ages, reverse=True)


class TestAggregate:
    def test_count_by_group(self):
        result = _apply_aggregate(SAMPLE_DATA, {"group_by": "dept", "agg": "count"})
        eng = next(r for r in result if r["dept"] == "Engineering")
        assert eng["count"] == 2

    def test_sum_by_group(self):
        result = _apply_aggregate(
            SAMPLE_DATA,
            {"group_by": "dept", "agg": "sum", "value_column": "age"},
        )
        eng = next(r for r in result if r["dept"] == "Engineering")
        assert eng["sum_age"] == 65.0


class TestDataTransformTool:
    def test_factory_returns_tool(self):
        tools = create_data_transform_tools()
        assert len(tools) == 1
        assert tools[0].name == "data_transform"

    @pytest.mark.asyncio
    async def test_csv_to_json_conversion(self, ctx):
        csv_file = ctx.workspace_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        json_file = ctx.workspace_path / "data.json"

        tool = DataTransformTool()
        result = await tool.execute({
            "input_file": "data.csv",
            "output_file": "data.json",
        }, ctx)

        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_filter_pipeline(self, ctx):
        json_file = ctx.workspace_path / "data.json"
        json_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")

        tool = DataTransformTool()
        result = await tool.execute({
            "input_file": "data.json",
            "operations": [
                {"type": "filter", "filter": {"column": "dept", "op": "==", "value": "Engineering"}},
            ],
        }, ctx)
        assert "Carol" in result
        assert "Bob" not in result

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self, ctx):
        tool = DataTransformTool()
        result = await tool.execute({"input_file": "missing.csv"}, ctx)
        assert "Error" in result
