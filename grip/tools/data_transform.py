"""Data transform tool — CSV/JSON/YAML pipeline operations.

Uses only stdlib (csv, json) plus optional yaml. Operations are applied
as a sequential pipeline: convert, filter, select, sort, aggregate.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from grip.tools.base import Tool, ToolContext

_SUPPORTED_EXTENSIONS = {".csv", ".json", ".yaml", ".yml"}


def _read_data(file_path: Path) -> list[dict[str, Any]]:
    """Read structured data from a file and return as list of dicts."""
    ext = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")

    if ext == ".csv":
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    elif ext == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return [{"value": data}]
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as err:
            raise ValueError(
                "PyYAML is required for YAML files. Install with: pip install pyyaml"
            ) from err
        data = yaml.safe_load(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return [{"value": data}]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _write_data(data: list[dict[str, Any]], file_path: Path) -> str:
    """Write structured data to a file in the format matching the extension."""
    ext = file_path.suffix.lower()

    if ext == ".csv":
        if not data:
            file_path.write_text("", encoding="utf-8")
            return "Wrote empty CSV."
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
        file_path.write_text(output.getvalue(), encoding="utf-8")
    elif ext == ".json":
        file_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as err:
            raise ValueError("PyYAML is required for YAML output.") from err
        file_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {ext}")

    return f"Wrote {len(data)} records to {file_path.name}"


def _apply_filter(data: list[dict[str, Any]], filter_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter rows based on column, operator, and value."""
    column = filter_spec.get("column", "")
    op = filter_spec.get("op", "==")
    value = filter_spec.get("value")

    if not column:
        return data

    result = []
    for row in data:
        cell = row.get(column)
        if cell is None:
            continue
        try:
            if op == "==":
                if str(cell) == str(value):
                    result.append(row)
            elif op == "!=":
                if str(cell) != str(value):
                    result.append(row)
            elif op == ">":
                if float(cell) > float(value):
                    result.append(row)
            elif op == "<":
                if float(cell) < float(value):
                    result.append(row)
            elif op == ">=":
                if float(cell) >= float(value):
                    result.append(row)
            elif op == "<=":
                if float(cell) <= float(value):
                    result.append(row)
            elif op == "contains" and str(value).lower() in str(cell).lower():
                result.append(row)
        except (ValueError, TypeError):
            continue
    return result


def _apply_select(data: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    """Select only specified columns from each row."""
    return [{col: row.get(col) for col in columns if col in row} for row in data]


def _apply_sort(data: list[dict[str, Any]], sort_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Sort data by a column."""
    by = sort_spec.get("by", "")
    reverse = sort_spec.get("reverse", False)
    if not by:
        return data

    def sort_key(row: dict) -> Any:
        val = row.get(by, "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val).lower()

    return sorted(data, key=sort_key, reverse=reverse)


def _apply_aggregate(data: list[dict[str, Any]], agg_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Group by a column and aggregate with count, sum, avg, min, max."""
    group_by = agg_spec.get("group_by", "")
    agg_func = agg_spec.get("agg", "count")
    value_col = agg_spec.get("value_column", "")

    if not group_by:
        return data

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        key = str(row.get(group_by, ""))
        groups.setdefault(key, []).append(row)

    result = []
    for key, rows in sorted(groups.items()):
        entry: dict[str, Any] = {group_by: key}
        if agg_func == "count":
            entry["count"] = len(rows)
        elif agg_func in ("sum", "avg", "min", "max") and value_col:
            values = []
            for r in rows:
                try:
                    values.append(float(r.get(value_col, 0)))
                except (ValueError, TypeError):
                    continue
            if values:
                if agg_func == "sum":
                    entry[f"{agg_func}_{value_col}"] = sum(values)
                elif agg_func == "avg":
                    entry[f"{agg_func}_{value_col}"] = round(sum(values) / len(values), 2)
                elif agg_func == "min":
                    entry[f"{agg_func}_{value_col}"] = min(values)
                elif agg_func == "max":
                    entry[f"{agg_func}_{value_col}"] = max(values)
        else:
            entry["count"] = len(rows)
        result.append(entry)

    return result


class DataTransformTool(Tool):
    """CSV/JSON/YAML pipeline: convert, filter, select, sort, aggregate."""

    @property
    def name(self) -> str:
        return "data_transform"

    @property
    def description(self) -> str:
        return "Transform CSV/JSON/YAML data: convert, filter, select, sort, and aggregate."

    @property
    def category(self) -> str:
        return "general"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to the input data file (CSV, JSON, or YAML).",
                },
                "output_file": {
                    "type": "string",
                    "description": "Path for the output file. Extension determines format.",
                },
                "operations": {
                    "type": "array",
                    "description": "Pipeline of operations applied sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["filter", "select", "sort", "aggregate"],
                            },
                            "filter": {
                                "type": "object",
                                "description": "Filter spec: {column, op, value}. Ops: ==, !=, >, <, >=, <=, contains.",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to select.",
                            },
                            "sort": {
                                "type": "object",
                                "description": "Sort spec: {by, reverse}.",
                            },
                            "aggregate": {
                                "type": "object",
                                "description": "Aggregate spec: {group_by, agg, value_column}.",
                            },
                        },
                    },
                },
            },
            "required": ["input_file"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        input_path = Path(params.get("input_file", ""))
        if not input_path.is_absolute():
            input_path = ctx.workspace_path / input_path

        if ctx.restrict_to_workspace:
            try:
                input_path.resolve().relative_to(ctx.workspace_path.resolve())
            except ValueError:
                return "Error: input file is outside the workspace sandbox."

        if not input_path.exists():
            return f"Error: input file '{input_path}' does not exist."

        try:
            data = _read_data(input_path)
        except Exception as exc:
            return f"Error reading input file: {exc}"

        operations = params.get("operations", [])
        for op in operations:
            op_type = op.get("type", "")
            if op_type == "filter" and "filter" in op:
                data = _apply_filter(data, op["filter"])
            elif op_type == "select" and "columns" in op:
                data = _apply_select(data, op["columns"])
            elif op_type == "sort" and "sort" in op:
                data = _apply_sort(data, op["sort"])
            elif op_type == "aggregate" and "aggregate" in op:
                data = _apply_aggregate(data, op["aggregate"])

        output_file = params.get("output_file", "")
        if output_file:
            output_path = Path(output_file)
            if not output_path.is_absolute():
                output_path = ctx.workspace_path / output_path
            if ctx.restrict_to_workspace:
                try:
                    output_path.resolve().relative_to(ctx.workspace_path.resolve())
                except ValueError:
                    return "Error: output file is outside the workspace sandbox."
            try:
                msg = _write_data(data, output_path)
                return f"{msg}\n\nPreview (first 5 rows):\n{json.dumps(data[:5], indent=2, default=str)}"
            except Exception as exc:
                return f"Error writing output: {exc}"

        return f"Transformed {len(data)} records:\n{json.dumps(data[:20], indent=2, default=str)}"


def create_data_transform_tools() -> list[Tool]:
    """Factory function returning data transform tool instances."""
    return [DataTransformTool()]
