import contextlib

from loguru import logger

from grip.tools.base import Tool, ToolContext, ToolRegistry
from grip.tools.code_analysis import create_code_analysis_tools
from grip.tools.data_transform import create_data_transform_tools
from grip.tools.document_gen import create_document_gen_tools
from grip.tools.email_compose import create_email_compose_tools
from grip.tools.filesystem import create_filesystem_tools
from grip.tools.finance import create_finance_tools
from grip.tools.mcp import MCPManager
from grip.tools.message import create_message_tools
from grip.tools.research import create_research_tools
from grip.tools.scheduler import create_scheduler_tools
from grip.tools.shell import create_shell_tools
from grip.tools.spawn import SubagentManager, create_spawn_tools
from grip.tools.web import create_web_tools
from grip.tools.workflow import create_workflow_tools

__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "SubagentManager",
    "create_code_analysis_tools",
    "create_data_transform_tools",
    "create_document_gen_tools",
    "create_email_compose_tools",
    "create_filesystem_tools",
    "create_finance_tools",
    "create_message_tools",
    "create_research_tools",
    "create_scheduler_tools",
    "create_shell_tools",
    "create_spawn_tools",
    "create_web_tools",
    "create_workflow_tools",
]


def create_default_registry(
    *,
    workspace_path: str | None = None,
    subagent_manager: SubagentManager | None = None,
    message_callback: object | None = None,
    mcp_servers: dict | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    registry.register_many(create_filesystem_tools())
    registry.register_many(create_shell_tools())
    registry.register_many(create_web_tools())
    registry.register_many(create_message_tools(message_callback))
    registry.register_many(create_spawn_tools(subagent_manager))
    registry.register_many(create_finance_tools())
    registry.register_many(create_research_tools())
    registry.register_many(create_code_analysis_tools())
    registry.register_many(create_data_transform_tools())
    registry.register_many(create_document_gen_tools())
    registry.register_many(create_email_compose_tools())
    registry.register_many(create_scheduler_tools())
    registry.register_many(create_workflow_tools())

    if mcp_servers:
        mcp_manager = MCPManager()
        registry.mcp_manager = mcp_manager
        try:
            import asyncio
            import threading

            def _load_mcp():
                loop = asyncio.new_event_loop()
                mcp_manager._event_loop = loop
                try:
                    loop.run_until_complete(mcp_manager.connect_all(mcp_servers, registry))
                except BaseException as exc:
                    logger.warning("MCP server connection failed: {}", exc)
                    loop.close()
                    return
                # Keep the event loop running so MCP transport background tasks
                # (anyio task groups inside streamablehttp/sse contexts) stay alive.
                # MCPManager.shutdown() calls loop.call_soon_threadsafe(loop.stop)
                # to exit cleanly. The daemon flag ensures this thread doesn't
                # block process exit if shutdown() is never called.
                with contextlib.suppress(BaseException):
                    loop.run_forever()
                loop.close()

            thread = threading.Thread(target=_load_mcp, daemon=True)
            thread.start()
        except Exception as exc:
            logger.warning("Failed to start MCP loading thread: {}", exc)

    return registry
