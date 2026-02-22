"""Tests for the tool system: registry, context, built-in tools, and MCP handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from grip.tools import create_default_registry
from grip.tools.base import Tool, ToolContext, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyTool(Tool):
    """Minimal tool implementation for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        return f"echo: {params.get('msg', '')}"


# ---------------------------------------------------------------------------
# ToolRegistry core operations
# ---------------------------------------------------------------------------


class TestRegistryCore:
    def test_register_and_get(self):
        """Register a tool and retrieve it by name."""
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)

        assert "dummy" in registry
        assert registry.get("dummy") is tool
        assert len(registry) == 1

    def test_names(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        assert "dummy" in registry.names()

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        assert registry.unregister("dummy") is True
        assert registry.unregister("dummy") is False
        assert len(registry) == 0

    def test_get_definitions(self):
        registry = ToolRegistry()
        registry.register(DummyTool())
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "dummy"

    @pytest.mark.asyncio
    async def test_execute(self, tmp_path: Path):
        registry = ToolRegistry()
        registry.register(DummyTool())
        ctx = ToolContext(workspace_path=tmp_path)
        result = await registry.execute("dummy", {"msg": "hello"}, ctx)
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_execute_unknown(self, tmp_path: Path):
        registry = ToolRegistry()
        ctx = ToolContext(workspace_path=tmp_path)
        result = await registry.execute("nonexistent", {}, ctx)
        assert "Error: Unknown tool" in result


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_defaults(self):
        ctx = ToolContext(workspace_path=Path("/tmp"))
        assert ctx.restrict_to_workspace is False
        assert ctx.shell_timeout == 60
        assert ctx.session_key == ""


# ---------------------------------------------------------------------------
# Default registry (built-in + new tools)
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_has_core_builtins(self):
        """create_default_registry should return a registry with built-in tools."""
        registry = create_default_registry()
        assert len(registry) >= 5
        assert "exec" in registry
        assert "read_file" in registry
        assert "write_file" in registry
        assert "spawn" in registry
        assert "send_message" in registry

    def test_all_new_tools_registered(self):
        """Verify all 6 new tools are present in the default registry."""
        registry = create_default_registry()
        expected_new_tools = {
            "web_research",
            "code_analysis",
            "data_transform",
            "document_gen",
            "email_compose",
            "scheduler",
        }
        registered = set(registry.names())
        for tool_name in expected_new_tools:
            assert tool_name in registered, f"Tool '{tool_name}' not registered"

    def test_existing_tools_still_registered(self):
        """Verify pre-existing tools are not broken by new registrations."""
        registry = create_default_registry()
        registered = set(registry.names())
        for tool_name in ("read_file", "write_file", "exec", "web_search", "web_fetch"):
            assert tool_name in registered, f"Existing tool '{tool_name}' missing"

    def test_all_definitions_valid(self):
        """Every registered tool should produce a valid OpenAI function-calling definition."""
        registry = create_default_registry()
        definitions = registry.get_definitions()
        assert len(definitions) == len(registry)
        for defn in definitions:
            assert defn["type"] == "function"
            assert "name" in defn["function"]
            assert "description" in defn["function"]
            assert "parameters" in defn["function"]


# ---------------------------------------------------------------------------
# MCP failure handling
# ---------------------------------------------------------------------------


class TestMCPFailureHandling:
    def test_registry_returned_despite_mcp_failure(self):
        """create_default_registry should return a valid registry even when MCP fails."""
        mock_server = MagicMock()
        mock_server.url = "http://broken:9999"
        mock_server.command = ""
        mock_server.headers = {}
        mock_server.args = []
        mock_server.env = {}

        registry = create_default_registry(mcp_servers={"broken": mock_server})
        assert registry is not None
        assert len(registry) > 0

    def test_mcp_failure_does_not_crash(self):
        """Broken MCP config should not prevent registry creation."""
        registry = create_default_registry(mcp_servers={"bad": MagicMock()})
        assert registry is not None
