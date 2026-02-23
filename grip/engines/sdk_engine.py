"""SDKRunner — EngineProtocol implementation using claude_agent_sdk.query().

This is the PRIMARY engine for Claude models. It delegates all tool execution,
agentic looping, and context management to the Claude Agent SDK. Grip handles:
  - System prompt assembly (identity files, memory, skills)
  - Custom tools (send_message, send_file, remember, recall)
  - MCP server config translation from grip format to SDK format
  - History persistence via MemoryManager
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from loguru import logger

from grip.engines.sdk_hooks import (
    build_post_tool_use_hook,
    build_pre_tool_use_hook,
    build_stop_hook,
)
from grip.engines.types import AgentRunResult, EngineProtocol
from grip.skills.loader import SkillsLoader

if TYPE_CHECKING:
    from grip.config.schema import GripConfig
    from grip.memory import MemoryManager
    from grip.memory.knowledge_base import KnowledgeBase
    from grip.session import SessionManager
    from grip.trust import TrustManager
    from grip.workspace import WorkspaceManager


class SDKRunner(EngineProtocol):
    """EngineProtocol implementation that uses claude_agent_sdk.query() for agentic runs.

    Unlike LiteLLMRunner (which wraps the internal AgentLoop), SDKRunner delegates
    the full agent loop to the Claude Agent SDK. Grip only provides the system
    prompt, custom tools, and MCP server configuration.
    """

    def __init__(
        self,
        config: GripConfig,
        workspace: WorkspaceManager,
        session_mgr: SessionManager,
        memory_mgr: MemoryManager,
        trust_mgr: TrustManager | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._session_mgr = session_mgr
        self._memory_mgr = memory_mgr
        self._trust_mgr = trust_mgr
        self._kb = knowledge_base

        # Resolve ANTHROPIC_API_KEY: config providers take priority, then env var.
        # Store it privately instead of writing to os.environ to prevent
        # exfiltration via child processes or shell commands.
        self._api_key = ""
        anthropic_provider = config.providers.get("anthropic")
        if anthropic_provider:
            self._api_key = anthropic_provider.api_key.get_secret_value()
        if not self._api_key:
            self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        defaults = config.agents.defaults
        self._model: str = defaults.sdk_model
        self._permission_mode: str = defaults.sdk_permission_mode
        self._cwd: str = str(workspace.root)
        self._mcp_servers = config.tools.mcp_servers
        self._send_callback: Callable | None = None
        self._send_file_callback: Callable | None = None

    # -- Callback wiring (called by gateway to route messages to channels) --

    def set_send_callback(self, callback: Callable) -> None:
        """Register the callback for send_message tool invocations."""
        self._send_callback = callback

    def set_send_file_callback(self, callback: Callable) -> None:
        """Register the callback for send_file tool invocations."""
        self._send_file_callback = callback

    # -- MCP config translation --

    def _build_mcp_config(self) -> list[dict[str, Any]]:
        """Convert grip MCPServerConfig entries to SDK-compatible dicts.

        Skips disabled servers. URL-based servers produce:
        {"name": ..., "url": ..., "headers": ..., "type": ...}
        Stdio-based servers produce:
        {"name": ..., "command": ..., "args": ..., "env": ...}
        """
        result: list[dict[str, Any]] = []
        for name, srv in self._mcp_servers.items():
            if not srv.enabled:
                continue
            if srv.url:
                entry: dict[str, Any] = {
                    "name": name,
                    "url": srv.url,
                    "headers": dict(srv.headers),
                }
                if srv.type:
                    entry["type"] = srv.type
                result.append(entry)
            elif srv.command:
                result.append(
                    {
                        "name": name,
                        "command": srv.command,
                        "args": list(srv.args),
                        "env": dict(srv.env),
                    }
                )
        return result

    def _collect_allowed_tools(self) -> list[str]:
        """Merge allowed_tools from all enabled MCP servers into a flat list."""
        tools: list[str] = []
        for _name, srv in self._mcp_servers.items():
            if not srv.enabled:
                continue
            tools.extend(srv.allowed_tools)
        return tools

    # -- System prompt assembly --

    def _build_system_prompt(self, user_message: str, session_key: str) -> str:
        """Assemble the system prompt from identity files, memory, skills, and metadata.

        Parts are joined with markdown horizontal rules for clear separation.
        Missing identity files are silently skipped.
        """
        parts: list[str] = []

        # Load identity files (AGENT.md, IDENTITY.md, SOUL.md, USER.md)
        identity_files = self._workspace.read_identity_files()
        for filename, content in identity_files.items():
            parts.append(f"## {filename}\n\n{content}")

        # Search long-term memory for relevant facts
        memory_results = self._memory_mgr.search_memory(user_message, max_results=5)
        if memory_results:
            memory_text = "\n".join(f"- {fact}" for fact in memory_results)
            parts.append(f"## Relevant Memory\n\n{memory_text}")

        # Search conversation history for relevant past interactions
        history_results = self._memory_mgr.search_history(user_message, max_results=5)
        if history_results:
            history_text = "\n".join(f"- {entry}" for entry in history_results)
            parts.append(f"## Relevant History\n\n{history_text}")

        # Inject learned behavioral patterns from KnowledgeBase (≤800 chars)
        if self._kb and self._kb.count > 0:
            kb_context = self._kb.export_for_context(max_chars=800)
            if kb_context:
                parts.append(f"## Learned Patterns\n\n{kb_context}")

        # Load available skills and list their names + descriptions
        try:
            loader = SkillsLoader(self._workspace.root)
            skills = loader.scan()
            if skills:
                skill_lines = [f"- **{s.name}**: {s.description}" for s in skills]
                parts.append("## Available Skills\n\n" + "\n".join(skill_lines))
        except Exception as exc:
            logger.debug("Failed to load skills for system prompt: {}", exc)

        # Runtime metadata
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        metadata = (
            f"## Runtime Metadata\n\n"
            f"- **Date/Time**: {now}\n"
            f"- **Session**: {session_key}\n"
            f"- **Workspace**: {self._cwd}"
        )
        parts.append(metadata)

        return "\n\n---\n\n".join(parts)

    # -- Custom tool definitions --

    @staticmethod
    def _text_result(text: str) -> dict[str, Any]:
        """Format a plain-text string as an MCP tool result."""
        return {"content": [{"type": "text", "text": text}]}

    def _build_custom_tools(self) -> list:
        """Build the list of custom tool functions for the SDK agent.

        Returns decorated callables that the SDK will expose as tools:
          - send_message: Route a text message through the gateway callback
          - send_file: Route a file through the gateway callback
          - remember: Store a fact in long-term memory
          - recall: Search long-term memory for matching facts
          - stock_quote: (optional) Fetch stock price if yfinance is installed

        Each tool uses the claude_agent_sdk @tool(name, description, input_schema)
        decorator and receives a single ``args`` dict parameter.
        """
        tools: list = []

        memory_mgr = self._memory_mgr
        runner = self

        @tool(
            "send_message",
            "Send a text message to the user via the configured channel.",
            {"text": str, "session_key": str},
        )
        async def send_message(args: dict[str, Any]) -> dict[str, Any]:
            cb = runner._send_callback
            if cb is None:
                return runner._text_result("Send callback not configured; message not delivered.")
            if inspect.iscoroutinefunction(cb):
                result = await cb(args["session_key"], args["text"])
            else:
                result = await asyncio.to_thread(cb, args["session_key"], args["text"])
            return runner._text_result(str(result))

        @tool(
            "send_file",
            "Send a file to the user via the configured channel.",
            {"file_path": str, "caption": str, "session_key": str},
        )
        async def send_file(args: dict[str, Any]) -> dict[str, Any]:
            cb = runner._send_file_callback
            if cb is None:
                return runner._text_result("Send file callback not configured; file not delivered.")
            if inspect.iscoroutinefunction(cb):
                result = await cb(args["session_key"], args["file_path"], args["caption"])
            else:
                result = await asyncio.to_thread(cb, args["session_key"], args["file_path"], args["caption"])
            return runner._text_result(str(result))

        @tool(
            "remember",
            "Store a fact in long-term memory for future recall.",
            {"fact": str, "category": str},
        )
        async def remember(args: dict[str, Any]) -> dict[str, Any]:
            entry = f"- [{args['category']}] {args['fact']}"
            memory_mgr.append_to_memory(entry)
            return runner._text_result(f"Stored fact under category '{args['category']}'.")

        @tool(
            "recall",
            "Search long-term memory for facts matching the query.",
            {"query_text": str},
        )
        async def recall(args: dict[str, Any]) -> dict[str, Any]:
            results = memory_mgr.search_memory(args["query_text"], max_results=10)
            if not results:
                return runner._text_result("No matching facts found in memory.")
            return runner._text_result("\n".join(results))

        tools.extend([send_message, send_file, remember, recall])

        try:
            import yfinance  # noqa: F401

            @tool(
                "stock_quote",
                "Fetch the current stock price for a given ticker symbol.",
                {"symbol": str},
            )
            async def stock_quote(args: dict[str, Any]) -> dict[str, Any]:
                import yfinance as yf

                def _fetch_quote(symbol: str) -> dict:
                    ticker = yf.Ticker(symbol)
                    return ticker.info

                info = await asyncio.to_thread(_fetch_quote, args["symbol"])
                price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
                name = info.get("shortName", args["symbol"])
                return runner._text_result(f"{name} ({args['symbol']}): ${price}")

            tools.append(stock_quote)
        except ImportError:
            pass

        return tools

    # -- EngineProtocol implementation --

    async def run(
        self,
        user_message: str,
        *,
        session_key: str = "cli:default",
        model: str | None = None,
    ) -> AgentRunResult:
        """Send a user message through the Claude Agent SDK and return the result.

        Streams messages from claude_agent_sdk.query(), collecting assistant text
        and tool call names. Persists the exchange to history via MemoryManager.
        """
        system_prompt = self._build_system_prompt(user_message, session_key)
        custom_tools = self._build_custom_tools()
        mcp_config = self._build_mcp_config()
        allowed_tools = self._collect_allowed_tools()

        effective_model = model or self._model

        # Wrap custom tools in an in-process MCP server (the SDK expects
        # ClaudeAgentOptions.tools to be a list of built-in tool name strings,
        # NOT SdkMcpTool objects).
        grip_server = create_sdk_mcp_server(
            "grip_tools", version="1.0.0", tools=custom_tools
        )
        mcp_servers: dict[str, Any] = {srv["name"]: srv for srv in mcp_config}
        mcp_servers["grip_tools"] = grip_server

        # Ensure the custom tool names are in allowed_tools so the SDK permits them
        custom_tool_names = [t.name for t in custom_tools]
        allowed_tools.extend(custom_tool_names)

        pre_hook = build_pre_tool_use_hook(Path(self._cwd), self._trust_mgr)
        post_hook = build_post_tool_use_hook()
        stop_hook = build_stop_hook(self._memory_mgr)

        env_opts: dict[str, str] = {}
        if self._api_key:
            env_opts["ANTHROPIC_API_KEY"] = self._api_key
        tool_search = self._config.tools.enable_tool_search
        if tool_search and tool_search != "auto":
            env_opts["ENABLE_TOOL_SEARCH"] = tool_search

        options = ClaudeAgentOptions(
            model=effective_model,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            permission_mode=self._permission_mode,
            cwd=self._cwd,
            allowed_tools=allowed_tools if allowed_tools else None,
            env=env_opts if env_opts else None,
            hooks={
                "pre_tool_use": pre_hook,
                "post_tool_use": post_hook,
                "stop": stop_hook,
            },
        )

        response_parts: list[str] = []
        tool_calls_made: list[str] = []

        async for message in query(prompt=user_message, options=options):
            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if hasattr(block, "text"):
                        response_parts.append(block.text)
                    if hasattr(block, "name"):
                        tool_calls_made.append(block.name)
            elif isinstance(message, ResultMessage):
                for block in getattr(message, "content", []):
                    if hasattr(block, "text"):
                        response_parts.append(block.text)

        response_text = "\n".join(response_parts) if response_parts else ""

        # Persist user message and agent response to conversation history
        self._memory_mgr.append_history(f"User ({session_key}): {user_message[:200]}")
        self._memory_mgr.append_history(f"Agent ({session_key}): {response_text[:200]}")

        return AgentRunResult(
            response=response_text,
            tool_calls_made=tool_calls_made,
        )

    async def consolidate_session(self, session_key: str) -> None:
        """No-op for SDK engine: the SDK manages its own context window internally.

        Logged for observability so operators can see when consolidation was requested.
        """
        logger.info(
            "consolidate_session called for '{}' (SDK handles context internally)", session_key
        )

    async def reset_session(self, session_key: str) -> None:
        """Clear all state for a session and delete persisted session."""
        self._session_mgr.delete(session_key)
        logger.info("Reset session '{}'", session_key)
