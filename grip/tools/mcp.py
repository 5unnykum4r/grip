"""MCP (Model Context Protocol) client integration.

Connects to MCP servers defined in config.tools.mcp_servers, discovers
their tools, and wraps them as grip Tool instances for the ToolRegistry.

Supports two transports:
  - stdio: spawn a subprocess (command + args)
  - HTTP/SSE: connect to a URL endpoint

Falls back gracefully if the mcp package is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from loguru import logger

from grip.config.schema import MCPServerConfig
from grip.tools.base import Tool, ToolContext, ToolRegistry


class MCPWrappedTool(Tool):
    """A grip Tool that delegates execution to an MCP server tool.

    Category defaults to 'mcp' but can be overridden per-server.
    """

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        server_name: str,
        call_fn: Any,
        mcp_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._name = f"mcp_{server_name}_{tool_name}"
        self._raw_name = tool_name
        self._description = tool_description
        self._parameters = tool_schema
        self._server_name = server_name
        self._call_fn = call_fn
        self._mcp_loop = mcp_loop

    @property
    def category(self) -> str:
        return "mcp"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"[MCP:{self._server_name}] {self._description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        try:
            coro = self._call_fn(self._raw_name, params)
            # The MCP session lives on a background thread with its own event
            # loop. Schedule the coroutine there and await the future from the
            # main loop to avoid cross-loop deadlocks.
            if self._mcp_loop is not None and self._mcp_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, self._mcp_loop)
                result = await asyncio.wrap_future(future)
            else:
                result = await coro
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return f"Error calling MCP tool '{self._raw_name}' on '{self._server_name}': {exc}"


class MCPConnection:
    """Manages a single MCP server connection (stdio or HTTP)."""

    def __init__(
        self,
        server_name: str,
        config: MCPServerConfig,
        *,
        force_oauth: bool = False,
    ) -> None:
        self.server_name = server_name
        self._config = config
        self._force_oauth = force_oauth
        self._session = None
        self._transport_cm = None
        self._read_stream = None
        self._write_stream = None
        self._process = None
        self._tools: list[MCPWrappedTool] = []
        self._connected: bool = False
        self._error: str = ""
        self._mcp_loop: asyncio.AbstractEventLoop | None = None

    @property
    def tools(self) -> list[MCPWrappedTool]:
        return list(self._tools)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str:
        return self._error

    async def connect(self) -> list[MCPWrappedTool]:
        """Connect to the MCP server and discover available tools."""
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            logger.warning("MCP package not installed. Install with: pip install mcp>=1.0")
            return []

        if self._config.url:
            return await self._connect_http()
        elif self._config.command:
            if self._config.oauth:
                await self._ensure_oauth_token()
                if self._error:
                    return []
            return await self._connect_stdio()
        else:
            logger.error("MCP server '{}' has no command or url configured", self.server_name)
            return []

    async def _ensure_oauth_token(self) -> None:
        """Load OAuth token from store, refresh if expired, inject into headers."""
        from grip.security.token_store import TokenStore

        store = TokenStore()
        token = store.get(self.server_name)

        if token is None:
            self._connected = False
            self._error = "OAuth login required"
            logger.warning(
                "MCP '{}' requires OAuth login (no token found). "
                "Run: grip mcp login {}",
                self.server_name,
                self.server_name,
            )
            return

        if token.is_expired and token.refresh_token:
            try:
                from grip.security.oauth import OAuthFlow

                flow = OAuthFlow(self._config.oauth, self.server_name)
                token = await flow.refresh(token.refresh_token)
                store.save(self.server_name, token)
                logger.info("Refreshed OAuth token for MCP '{}'", self.server_name)
            except Exception as exc:
                self._connected = False
                self._error = f"Token refresh failed: {exc}"
                logger.error("Failed to refresh OAuth token for '{}': {}", self.server_name, exc)
                return
        elif token.is_expired:
            self._connected = False
            self._error = "OAuth token expired (no refresh token)"
            logger.warning("OAuth token expired for MCP '{}' with no refresh token", self.server_name)
            return

        self._config.headers["Authorization"] = f"Bearer {token.access_token}"

    async def _connect_stdio(self) -> list[MCPWrappedTool]:
        """Connect via stdio transport (spawn subprocess)."""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self._config.command,
                args=self._config.args,
                env=self._config.env if self._config.env else None,
            )

            self._transport_cm = stdio_client(params)
            self._read_stream, self._write_stream = await self._transport_cm.__aenter__()
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            await self._session.initialize()

            tools_response = await self._session.list_tools()
            self._connected = True
            self._error = ""
            return self._wrap_tools(tools_response.tools)

        except Exception as exc:
            self._connected = False
            self._error = str(exc)
            logger.error("Failed to connect stdio MCP '{}': {}", self.server_name, exc)
            return []

    async def _connect_http(self) -> list[MCPWrappedTool]:
        """Connect via HTTP (streamable) or SSE transport.

        Uses streamable HTTP transport when config.type == "http".
        Falls back to SSE transport for type == "sse" or unspecified.
        Automatically attaches OAuthClientProvider for servers that require
        OAuth (e.g. Supabase with dynamic client registration). The provider
        only activates on 401 responses — no-auth servers work normally.
        """
        try:
            from mcp import ClientSession

            headers = self._config.headers if self._config.headers else None
            timeout = float(self._config.timeout)

            oauth_auth = self._build_oauth_auth()

            if self._config.type == "http":
                from mcp.client.streamable_http import streamablehttp_client

                self._transport_cm = streamablehttp_client(
                    self._config.url,
                    headers=headers,
                    timeout=timeout,
                    auth=oauth_auth,
                )
                streams = await self._transport_cm.__aenter__()
                # streamablehttp_client yields (read, write, get_session_id)
                self._read_stream = streams[0]
                self._write_stream = streams[1]
            else:
                from mcp.client.sse import sse_client

                self._transport_cm = sse_client(
                    self._config.url,
                    headers=headers,
                    timeout=timeout,
                    auth=oauth_auth,
                )
                self._read_stream, self._write_stream = await self._transport_cm.__aenter__()

            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            await self._session.initialize()

            tools_response = await self._session.list_tools()
            self._connected = True
            self._error = ""
            return self._wrap_tools(tools_response.tools)

        except Exception as exc:
            self._connected = False
            exc_str = str(exc)
            if "401" in exc_str or "Unauthorized" in exc_str:
                self._error = "OAuth login required"
                logger.warning(
                    "MCP '{}' requires authentication. Run: /mcp → select '{}' → Login",
                    self.server_name,
                    self.server_name,
                )
            else:
                self._error = exc_str
                logger.error("Failed to connect HTTP MCP '{}': {}", self.server_name, exc)
            return []

    def _build_oauth_auth(self) -> Any:
        """Create an OAuthClientProvider for token refresh only, or None.

        Only attaches the OAuth provider when a stored **token** already
        exists (i.e. the user previously logged in via `/mcp` → Login or
        the gateway callback). A stored client registration alone is not
        enough — the OAuthClientProvider would try to open a browser for
        interactive authorization, which must not happen during background
        startup connections.
        """
        try:
            from grip.tools.mcp_auth import MCPTokenStorage, create_mcp_oauth_auth
        except ImportError:
            return None

        storage = MCPTokenStorage(self.server_name)
        has_token = storage.has_stored_token()

        if not has_token:
            return None

        callback_port = (
            self._config.oauth.redirect_port
            if self._config.oauth
            else 18801
        )
        return create_mcp_oauth_auth(
            server_name=self.server_name,
            server_url=self._config.url,
            callback_port=callback_port,
        )

    def _wrap_tools(self, mcp_tools) -> list[MCPWrappedTool]:
        """Convert MCP tool definitions into grip Tool wrappers."""
        self._tools = []
        for tool in mcp_tools:
            schema = (
                tool.inputSchema
                if hasattr(tool, "inputSchema")
                else {"type": "object", "properties": {}}
            )
            wrapped = MCPWrappedTool(
                tool_name=tool.name,
                tool_description=getattr(tool, "description", tool.name),
                tool_schema=schema,
                server_name=self.server_name,
                call_fn=self._call_tool,
                mcp_loop=self._mcp_loop,
            )
            self._tools.append(wrapped)
            logger.debug("Discovered MCP tool: {}.{}", self.server_name, tool.name)

        logger.info("MCP '{}': discovered {} tools", self.server_name, len(self._tools))
        return self._tools

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call on the connected MCP server."""
        if not self._session:
            raise RuntimeError(f"MCP session '{self.server_name}' not connected")
        result = await self._session.call_tool(tool_name, arguments)
        # MCP returns content blocks; extract text
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts) if texts else str(result.content)

    async def disconnect(self) -> None:
        """Close the MCP session and underlying transport."""
        if self._session:
            with contextlib.suppress(Exception):
                await self._session.__aexit__(None, None, None)
        if self._transport_cm:
            with contextlib.suppress(Exception):
                await self._transport_cm.__aexit__(None, None, None)
        logger.debug("MCP '{}' disconnected", self.server_name)


class MCPManager:
    """Manages all MCP server connections and their lifecycle."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._registry: ToolRegistry | None = None
        self._event_loop: Any = None

    async def connect_all(
        self,
        mcp_servers: dict[str, MCPServerConfig],
        registry: ToolRegistry,
    ) -> int:
        """Connect to all configured MCP servers and register their tools.

        Returns the total number of MCP tools registered.
        """
        self._registry = registry
        total_tools = 0

        # Capture the running event loop so wrapped tools can schedule
        # coroutines back onto it from the main thread.
        mcp_loop = asyncio.get_running_loop()

        for server_name, server_config in mcp_servers.items():
            if not server_config.enabled:
                logger.info("MCP '{}' is disabled, skipping", server_name)
                continue
            conn = MCPConnection(server_name, server_config)
            conn._mcp_loop = mcp_loop
            tools = await conn.connect()
            self._connections[server_name] = conn

            for tool in tools:
                registry.register(tool)
                total_tools += 1

        if total_tools:
            n = len(self._connections)
            label = "server" if n == 1 else "servers"
            logger.info("Discovered and registered {} tools from {} MCP {}", total_tools, n, label)
        return total_tools

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for conn in self._connections.values():
            await conn.disconnect()
        self._connections.clear()

    def shutdown(self) -> None:
        """Stop the background event loop that keeps MCP transports alive.

        Safe to call from any thread. The background thread created by
        create_default_registry() stores its loop as ``_event_loop``.
        Calling ``loop.call_soon_threadsafe(loop.stop)`` causes
        ``run_forever()`` to exit, and the thread cleans up.
        """
        loop = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    @property
    def server_count(self) -> int:
        return len(self._connections)

    async def reconnect_server(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        registry: ToolRegistry | None = None,
    ) -> list[MCPWrappedTool]:
        """Disconnect an existing connection (if any), reconnect, and register tools.

        Uses the stored registry from connect_all() if none is provided.
        Returns the list of newly registered tools.
        """
        reg = registry or self._registry
        if reg is None:
            logger.error("MCPManager.reconnect_server: no registry available")
            return []

        existing = self._connections.get(server_name)
        if existing:
            await existing.disconnect()
            for tool in existing.tools:
                reg.unregister(tool.name)

        conn = MCPConnection(server_name, server_config)
        conn._mcp_loop = self._event_loop
        tools = await conn.connect()
        self._connections[server_name] = conn

        for tool in tools:
            reg.register(tool)

        return tools

    def get_connection(self, server_name: str) -> MCPConnection | None:
        return self._connections.get(server_name)
