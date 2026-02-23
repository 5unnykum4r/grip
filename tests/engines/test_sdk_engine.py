"""Tests for SDKRunner — the EngineProtocol implementation using claude_agent_sdk."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from grip.config.schema import GripConfig
from grip.engines.types import AgentRunResult, EngineProtocol

# ---------------------------------------------------------------------------
# Mock the claude_agent_sdk module before importing SDKRunner, since the real
# SDK package may not be installed in the test environment.
# ---------------------------------------------------------------------------

_mock_sdk = types.ModuleType("claude_agent_sdk")
_mock_sdk.query = MagicMock(name="query")


def _mock_tool_decorator(name: str, description: str, input_schema):
    """Mock for claude_agent_sdk.tool(name, description, input_schema).

    Returns a decorator that attaches name/description metadata to the
    function and returns it unchanged so tests can call it directly.
    """

    def decorator(fn):
        fn._tool_name = name
        fn._tool_description = description
        fn._tool_input_schema = input_schema
        return fn

    return decorator


_mock_sdk.tool = _mock_tool_decorator
_mock_sdk.ClaudeAgentOptions = MagicMock(name="ClaudeAgentOptions")
_mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
_mock_sdk.ResultMessage = type("ResultMessage", (), {})


@pytest.fixture(autouse=True)
def _install_mock_sdk(monkeypatch):
    """Insert the mock claude_agent_sdk into sys.modules for every test."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _mock_sdk)
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace directory structure."""
    ws = tmp_path / "workspace"
    for dirname in ("sessions", "memory", "skills", "cron", "hooks", "workflows", "state"):
        (ws / dirname).mkdir(parents=True)
    (ws / "AGENT.md").write_text("# Test Agent\nYou are a test agent.")
    (ws / "IDENTITY.md").write_text("# Identity\nName: TestBot")
    (ws / "SOUL.md").write_text("# Soul\nBe helpful.")
    (ws / "USER.md").write_text("# User\nName: Tester")
    (ws / "memory" / "MEMORY.md").write_text("")
    (ws / "memory" / "HISTORY.md").write_text("")
    return ws


@pytest.fixture
def config(tmp_workspace: Path) -> GripConfig:
    """Minimal test config pointing at the temporary workspace."""
    return GripConfig(
        agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
        tools={"restrict_to_workspace": True},
        providers={"anthropic": {"api_key": "test-key-12345"}},
    )


@pytest.fixture
def config_no_api_key(tmp_workspace: Path) -> GripConfig:
    """Config with no anthropic API key (tests env fallback)."""
    return GripConfig(
        agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
        tools={"restrict_to_workspace": True},
    )


@pytest.fixture
def config_with_mcp(tmp_workspace: Path) -> GripConfig:
    """Config with MCP server definitions for testing _build_mcp_config."""
    return GripConfig(
        agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
        tools={
            "restrict_to_workspace": True,
            "mcp_servers": {
                "http_server": {
                    "url": "https://mcp.example.com/sse",
                    "headers": {"Authorization": "Bearer tok"},
                },
                "stdio_server": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-fs"],
                    "env": {"HOME": "/tmp"},
                },
            },
        },
    )


@pytest.fixture
def mock_workspace(tmp_workspace: Path) -> MagicMock:
    ws = MagicMock()
    ws.root = tmp_workspace
    ws.is_initialized = True
    ws.read_file = MagicMock(side_effect=lambda name: _read_ws_file(tmp_workspace, name))
    ws.read_identity_files = MagicMock(
        return_value={
            "AGENT.md": "# Test Agent\nYou are a test agent.",
            "IDENTITY.md": "# Identity\nName: TestBot",
            "SOUL.md": "# Soul\nBe helpful.",
            "USER.md": "# User\nName: Tester",
        }
    )
    return ws


def _read_ws_file(ws_path: Path, name: str) -> str | None:
    target = ws_path / name
    if target.is_file():
        return target.read_text(encoding="utf-8")
    return None


@pytest.fixture
def mock_session_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.get_or_create = MagicMock()
    mgr.delete = MagicMock(return_value=True)
    mgr.save = MagicMock()
    return mgr


@pytest.fixture
def mock_memory_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.search_memory = MagicMock(return_value=["fact: user prefers dark mode"])
    mgr.search_history = MagicMock(return_value=["[2025-01-01] discussed project setup"])
    mgr.append_to_memory = MagicMock()
    mgr.append_history = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# Helper: build an SDKRunner with mocked dependencies
# ---------------------------------------------------------------------------


def _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr, trust_mgr=None):
    """Construct an SDKRunner with mocked claude_agent_sdk."""
    from grip.engines.sdk_engine import SDKRunner

    runner = SDKRunner(
        config=config,
        workspace=mock_workspace,
        session_mgr=mock_session_mgr,
        memory_mgr=mock_memory_mgr,
        trust_mgr=trust_mgr,
    )
    return runner


def _find_tool(tools: list, name: str):
    """Find a tool function by its _tool_name attribute or __name__."""
    for fn in tools:
        tool_name = getattr(fn, "_tool_name", None) or getattr(fn, "__name__", "")
        if tool_name == name:
            return fn
    raise KeyError(
        f"Tool '{name}' not found in {[getattr(t, '_tool_name', t.__name__) for t in tools]}"
    )


# ---------------------------------------------------------------------------
# Tests: Protocol conformance
# ---------------------------------------------------------------------------


class TestSDKRunnerIsEngineProtocol:
    """SDKRunner must satisfy the EngineProtocol ABC."""

    def test_is_subclass(self):
        from grip.engines.sdk_engine import SDKRunner

        assert issubclass(SDKRunner, EngineProtocol)

    def test_isinstance_check(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert isinstance(runner, EngineProtocol)


# ---------------------------------------------------------------------------
# Tests: Constructor
# ---------------------------------------------------------------------------


class TestSDKRunnerConstructor:
    """Constructor should set up all fields correctly."""

    def test_sets_model_from_config(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._model == "claude-sonnet-4-6"

    def test_sets_permission_mode(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._permission_mode == "acceptEdits"

    def test_sets_cwd_to_workspace_path(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._cwd == str(mock_workspace.root)

    def test_initializes_callbacks_to_none(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._send_callback is None
        assert runner._send_file_callback is None

    def test_resolves_api_key_from_config(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._api_key == "test-key-12345"
        # API key should NOT be leaked into os.environ
        assert os.environ.get("ANTHROPIC_API_KEY") != "test-key-12345"

    def test_falls_back_to_env_api_key(
        self, config_no_api_key, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key-67890"}):
            runner = _build_runner(config_no_api_key, mock_workspace, mock_session_mgr, mock_memory_mgr)
            assert runner._api_key == "env-key-67890"

    def test_stores_trust_manager(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        fake_trust = MagicMock(name="trust_mgr")
        runner = _build_runner(
            config, mock_workspace, mock_session_mgr, mock_memory_mgr, trust_mgr=fake_trust
        )
        assert runner._trust_mgr is fake_trust


# ---------------------------------------------------------------------------
# Tests: Callbacks
# ---------------------------------------------------------------------------


class TestSDKRunnerCallbacks:
    """set_send_callback and set_send_file_callback should store callables."""

    def test_set_send_callback(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        callback = MagicMock()
        runner.set_send_callback(callback)
        assert runner._send_callback is callback

    def test_set_send_file_callback(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        callback = MagicMock()
        runner.set_send_file_callback(callback)
        assert runner._send_file_callback is callback


# ---------------------------------------------------------------------------
# Tests: _build_mcp_config
# ---------------------------------------------------------------------------


class TestBuildMCPConfig:
    """_build_mcp_config should convert grip MCPServerConfig to SDK format."""

    def test_empty_mcp_servers(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        assert result == []

    def test_url_based_server(
        self, config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()

        url_entries = [e for e in result if "url" in e]
        assert len(url_entries) == 1
        entry = url_entries[0]
        assert entry["name"] == "http_server"
        assert entry["url"] == "https://mcp.example.com/sse"
        assert entry["headers"] == {"Authorization": "Bearer tok"}

    def test_stdio_based_server(
        self, config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()

        stdio_entries = [e for e in result if "command" in e]
        assert len(stdio_entries) == 1
        entry = stdio_entries[0]
        assert entry["name"] == "stdio_server"
        assert entry["command"] == "npx"
        assert entry["args"] == ["-y", "@modelcontextprotocol/server-fs"]
        assert entry["env"] == {"HOME": "/tmp"}

    def test_returns_both_types(
        self, config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        assert len(result) == 2

    def test_skips_disabled_servers(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "enabled_server": {"command": "npx", "args": ["-y", "test"]},
                    "disabled_server": {"command": "npx", "args": ["-y", "off"], "enabled": False},
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        names = [e["name"] for e in result]
        assert "enabled_server" in names
        assert "disabled_server" not in names

    def test_passes_type_for_url_server(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "sse_server": {
                        "url": "https://mcp.example.com/sse",
                        "type": "sse",
                    },
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        assert len(result) == 1
        assert result[0]["type"] == "sse"

    def test_omits_type_when_empty(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "plain": {"url": "https://mcp.example.com"},
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        assert len(result) == 1
        assert "type" not in result[0]

    def test_supabase_http_type_passthrough(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "supabase": {
                        "url": "https://mcp.supabase.com/mcp",
                        "type": "http",
                    },
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        result = runner._build_mcp_config()
        assert len(result) == 1
        assert result[0]["name"] == "supabase"
        assert result[0]["url"] == "https://mcp.supabase.com/mcp"
        assert result[0]["type"] == "http"


# ---------------------------------------------------------------------------
# Tests: _collect_allowed_tools
# ---------------------------------------------------------------------------


class TestCollectAllowedTools:
    def test_empty_when_no_servers(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._collect_allowed_tools() == []

    def test_merges_from_multiple_servers(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "s1": {
                        "command": "npx",
                        "args": [],
                        "allowed_tools": ["mcp__s1__read", "mcp__s1__write"],
                    },
                    "s2": {
                        "url": "https://s2.example.com",
                        "allowed_tools": ["mcp__s2__*"],
                    },
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._collect_allowed_tools()
        assert "mcp__s1__read" in tools
        assert "mcp__s1__write" in tools
        assert "mcp__s2__*" in tools
        assert len(tools) == 3

    def test_skips_disabled_servers(
        self, tmp_workspace, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        config = GripConfig(
            agents={"defaults": {"workspace": str(tmp_workspace), "semantic_cache_enabled": False}},
            tools={
                "restrict_to_workspace": True,
                "mcp_servers": {
                    "active": {"command": "npx", "args": [], "allowed_tools": ["mcp__active__*"]},
                    "off": {"command": "npx", "args": [], "allowed_tools": ["mcp__off__*"], "enabled": False},
                },
            },
        )
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._collect_allowed_tools()
        assert "mcp__active__*" in tools
        assert "mcp__off__*" not in tools

    def test_empty_allowed_tools_produces_empty(
        self, config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config_with_mcp, mock_workspace, mock_session_mgr, mock_memory_mgr)
        assert runner._collect_allowed_tools() == []


# ---------------------------------------------------------------------------
# Tests: _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """_build_system_prompt should assemble identity files, memory, and skills."""

    def test_includes_identity_files(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        assert "AGENT.md" in prompt
        assert "IDENTITY.md" in prompt
        assert "SOUL.md" in prompt
        assert "USER.md" in prompt

    def test_includes_memory_search_results(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        mock_memory_mgr.search_memory.assert_called_once_with("hello", max_results=5)
        assert "user prefers dark mode" in prompt

    def test_includes_history_search_results(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        mock_memory_mgr.search_history.assert_called_once_with("hello", max_results=5)
        assert "discussed project setup" in prompt

    def test_includes_runtime_metadata(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        assert "test:session" in prompt
        assert str(mock_workspace.root) in prompt

    def test_parts_joined_with_separator(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        assert "\n\n---\n\n" in prompt

    def test_handles_missing_identity_files(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        """When workspace returns no identity files, prompt still has memory and metadata."""
        mock_workspace.read_identity_files.return_value = {}
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        assert "test:session" in prompt

    def test_handles_empty_memory_results(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        mock_memory_mgr.search_memory.return_value = []
        mock_memory_mgr.search_history.return_value = []
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        prompt = runner._build_system_prompt("hello", "test:session")

        assert "AGENT.md" in prompt


# ---------------------------------------------------------------------------
# Tests: _build_custom_tools
# ---------------------------------------------------------------------------


class TestBuildCustomTools:
    """_build_custom_tools should return a list of callable tool functions."""

    def test_returns_list(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        assert isinstance(tools, list)

    def test_minimum_tool_count(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        """At minimum: send_message, send_file, remember, recall."""
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        assert len(tools) >= 4

    def test_tool_functions_are_callable(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        for tool_fn in tools:
            assert callable(tool_fn)

    def test_tool_names(self, config, mock_workspace, mock_session_mgr, mock_memory_mgr):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        tool_names = {getattr(fn, "_tool_name", fn.__name__) for fn in tools}
        assert "send_message" in tool_names
        assert "send_file" in tool_names
        assert "remember" in tool_names
        assert "recall" in tool_names

    @pytest.mark.asyncio
    async def test_remember_tool_calls_memory_manager(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        remember_fn = _find_tool(tools, "remember")
        await remember_fn({"fact": "user likes Python", "category": "preferences"})
        mock_memory_mgr.append_to_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_tool_calls_memory_search(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        recall_fn = _find_tool(tools, "recall")
        await recall_fn({"query_text": "Python"})
        mock_memory_mgr.search_memory.assert_called_with("Python", max_results=10)

    @pytest.mark.asyncio
    async def test_send_message_invokes_callback(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        callback = MagicMock(return_value="sent")
        runner.set_send_callback(callback)

        tools = runner._build_custom_tools()
        send_fn = _find_tool(tools, "send_message")
        await send_fn({"text": "Hello world", "session_key": "test:session"})
        callback.assert_called_once_with("test:session", "Hello world")

    @pytest.mark.asyncio
    async def test_send_message_without_callback(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        send_fn = _find_tool(tools, "send_message")
        result = await send_fn({"text": "Hello world", "session_key": "test:session"})
        text = result["content"][0]["text"]
        assert "not configured" in text.lower()

    @pytest.mark.asyncio
    async def test_send_file_invokes_callback(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        callback = MagicMock(return_value="file sent")
        runner.set_send_file_callback(callback)

        tools = runner._build_custom_tools()
        send_file_fn = _find_tool(tools, "send_file")
        await send_file_fn(
            {
                "file_path": "/path/to/file.txt",
                "caption": "a caption",
                "session_key": "test:session",
            }
        )
        callback.assert_called_once_with("test:session", "/path/to/file.txt", "a caption")

    @pytest.mark.asyncio
    async def test_send_file_without_callback(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        tools = runner._build_custom_tools()
        send_file_fn = _find_tool(tools, "send_file")
        result = await send_file_fn(
            {"file_path": "/path/to/file.txt", "caption": "cap", "session_key": "test:session"}
        )
        text = result["content"][0]["text"]
        assert "not configured" in text.lower()


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestSDKRunnerRun:
    """run() should call query() and collect messages into AgentRunResult."""

    @pytest.mark.asyncio
    async def test_run_returns_agent_run_result(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)

        assistant_msg = MagicMock()
        assistant_msg.__class__ = _mock_sdk.AssistantMessage
        text_block = MagicMock()
        text_block.text = "Hello from SDK"
        del text_block.name
        assistant_msg.content = [text_block]

        result_msg = MagicMock()
        result_msg.__class__ = _mock_sdk.ResultMessage
        result_text_block = MagicMock()
        result_text_block.text = "Final answer"
        del result_text_block.name
        result_msg.content = [result_text_block]

        async def mock_query_iter(**kwargs):
            yield assistant_msg
            yield result_msg

        with patch("grip.engines.sdk_engine.query", side_effect=mock_query_iter):
            result = await runner.run("test message", session_key="test:session")

        assert isinstance(result, AgentRunResult)
        assert "Final answer" in result.response

    @pytest.mark.asyncio
    async def test_run_collects_tool_names(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)

        assistant_msg = MagicMock()
        assistant_msg.__class__ = _mock_sdk.AssistantMessage
        tool_block = MagicMock()
        tool_block.name = "read_file"
        del tool_block.text
        assistant_msg.content = [tool_block]

        result_msg = MagicMock()
        result_msg.__class__ = _mock_sdk.ResultMessage
        result_text_block = MagicMock()
        result_text_block.text = "Done"
        del result_text_block.name
        result_msg.content = [result_text_block]

        async def mock_query_iter(**kwargs):
            yield assistant_msg
            yield result_msg

        with patch("grip.engines.sdk_engine.query", side_effect=mock_query_iter):
            result = await runner.run("test message", session_key="test:session")

        assert "read_file" in result.tool_calls_made

    @pytest.mark.asyncio
    async def test_run_persists_history(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)

        result_msg = MagicMock()
        result_msg.__class__ = _mock_sdk.ResultMessage
        result_text_block = MagicMock()
        result_text_block.text = "Response text"
        del result_text_block.name
        result_msg.content = [result_text_block]

        async def mock_query_iter(**kwargs):
            yield result_msg

        with patch("grip.engines.sdk_engine.query", side_effect=mock_query_iter):
            await runner.run("user question", session_key="test:session")

        assert mock_memory_mgr.append_history.call_count == 2

    @pytest.mark.asyncio
    async def test_run_uses_override_model(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)

        result_msg = MagicMock()
        result_msg.__class__ = _mock_sdk.ResultMessage
        result_text_block = MagicMock()
        result_text_block.text = "ok"
        del result_text_block.name
        result_msg.content = [result_text_block]

        captured_kwargs = {}

        async def mock_query_iter(**kwargs):
            captured_kwargs.update(kwargs)
            yield result_msg

        with patch("grip.engines.sdk_engine.query", side_effect=mock_query_iter):
            await runner.run("test", session_key="s", model="claude-opus-4-6")

        assert captured_kwargs.get("options") is not None


# ---------------------------------------------------------------------------
# Tests: Session management
# ---------------------------------------------------------------------------


class TestSDKRunnerSessionManagement:
    """consolidate_session and reset_session should handle sessions correctly."""

    @pytest.mark.asyncio
    async def test_consolidate_session_logs_only(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        """consolidate_session just logs (SDK handles context internally)."""
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        await runner.consolidate_session("test:session")

    @pytest.mark.asyncio
    async def test_reset_session_deletes_via_session_mgr(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        await runner.reset_session("test:session")
        mock_session_mgr.delete.assert_called_once_with("test:session")

    @pytest.mark.asyncio
    async def test_reset_session_handles_missing_client(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        """reset_session should not raise if the session key has no client."""
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)
        await runner.reset_session("nonexistent:session")
        mock_session_mgr.delete.assert_called_once_with("nonexistent:session")


# ---------------------------------------------------------------------------
# Tests: Skills loading in system prompt
# ---------------------------------------------------------------------------


class TestSDKRunnerSkillsInPrompt:
    """_build_system_prompt should include skill names and descriptions."""

    def test_includes_skills_section(
        self, config, mock_workspace, mock_session_mgr, mock_memory_mgr
    ):
        runner = _build_runner(config, mock_workspace, mock_session_mgr, mock_memory_mgr)

        mock_skill = MagicMock()
        mock_skill.name = "web_search"
        mock_skill.description = "Search the web for information"

        with patch("grip.engines.sdk_engine.SkillsLoader") as mock_loader:
            loader_instance = MagicMock()
            loader_instance.scan.return_value = [mock_skill]
            mock_loader.return_value = loader_instance

            prompt = runner._build_system_prompt("hello", "test:session")

        assert "web_search" in prompt
        assert "Search the web" in prompt
