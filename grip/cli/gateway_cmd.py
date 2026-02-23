"""grip gateway -- run the full agent platform with channels, cron, heartbeat, and API.

The gateway is the long-running process that connects:
  - REST API server (FastAPI + uvicorn)
  - Chat channels (Telegram, Discord, Slack) via the message bus
  - Cron scheduler for periodic tasks
  - Heartbeat service for autonomous check-ins
  - Engine (LiteLLM or Claude SDK) for processing all inbound messages

Start: grip gateway
Stop:  Ctrl+C (SIGINT) or SIGTERM for graceful shutdown
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

import typer
from loguru import logger
from rich.console import Console

from grip import __version__
from grip.bus.events import InboundMessage, OutboundMessage
from grip.bus.queue import MessageBus
from grip.channels.manager import ChannelManager
from grip.config import GripConfig, load_config
from grip.cron.service import CronService
from grip.engines.factory import create_engine
from grip.engines.types import EngineProtocol
from grip.heartbeat.service import HeartbeatService
from grip.memory import MemoryManager
from grip.session import SessionManager
from grip.tools.message import MessageTool, SendFileTool
from grip.trust import TrustManager
from grip.workspace import WorkspaceManager

console = Console()


def gateway_command(
    host: str = typer.Option(None, "--host", "-H", help="Bind address (overrides config)."),  # noqa: B008
    port: int = typer.Option(None, "--port", "-p", help="Bind port (overrides config)."),  # noqa: B008
) -> None:
    """Start the grip gateway: channels + engine + cron + heartbeat."""
    from grip.cli.app import state

    config = load_config(state.config_path)
    try:
        asyncio.run(_run_gateway(config, host=host, port=port))
    except KeyboardInterrupt:
        console.print("\n[dim]Gateway shutdown by user.[/dim]")


async def _run_gateway(
    config: GripConfig, host: str | None = None, port: int | None = None
) -> None:
    """Main async entry point for the gateway process."""
    if host:
        config.gateway.host = host
    if port:
        config.gateway.port = port
    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    ws = WorkspaceManager(ws_path)
    if not ws.is_initialized:
        ws.initialize()

    session_mgr = SessionManager(ws.root / "sessions")
    memory_mgr = MemoryManager(ws.root)
    trust_mgr = TrustManager(ws.root / "state")

    # Create the engine via the factory (reads config.agents.defaults.engine
    # to pick SDKRunner or LiteLLMRunner automatically)
    engine = create_engine(config, ws, session_mgr, memory_mgr, trust_mgr=trust_mgr)

    # Wire outbound messaging based on engine type
    bus = MessageBus()
    _wire_engine_messaging(engine, bus)

    # Start channels
    channel_mgr = ChannelManager(config.channels)
    started_channels = await channel_mgr.start_all(bus)
    if started_channels:
        console.print(f"[green]Channels started:[/green] {', '.join(started_channels)}")
    else:
        console.print(
            "[yellow]No channels enabled. Gateway will only process internal events.[/yellow]"
        )

    # Start cron and heartbeat (cron gets bus access for result delivery).
    # Both services accept Any for their agent_loop parameter and only call
    # .run() on it, which EngineProtocol satisfies.
    cron_svc = CronService(ws.root / "cron", engine, config.cron, bus=bus)
    heartbeat_svc = HeartbeatService(
        ws.root, engine, config.heartbeat,
        bus=bus, reply_to=config.heartbeat.reply_to,
    )

    if config.heartbeat.enabled and config.heartbeat.interval_minutes < 10:
        console.print(
            f"[yellow]Warning:[/yellow] Heartbeat interval is {config.heartbeat.interval_minutes} minutes. "
            "Each heartbeat triggers a full agent run consuming tokens. "
            "Consider increasing the interval to reduce costs."
        )

    cron_task = asyncio.create_task(cron_svc.start(), name="cron-service")
    heartbeat_task = asyncio.create_task(heartbeat_svc.start(), name="heartbeat-service")

    # Start REST API server if fastapi+uvicorn are installed
    api_task: asyncio.Task | None = None
    api_task = _start_api_server(config, engine, session_mgr, memory_mgr, ws, cron_svc)

    console.print("[bold cyan]grip gateway running.[/bold cyan] Press Ctrl+C to stop.")

    # Set up graceful shutdown via signals
    shutdown_event = asyncio.Event()
    _install_signal_handlers(shutdown_event)

    # Run the inbound message consumer alongside shutdown watcher
    consumer_task = asyncio.create_task(
        _consume_inbound(bus, engine, session_mgr, memory_mgr, config, trust_mgr),
        name="inbound-consumer",
    )

    await shutdown_event.wait()

    # Graceful shutdown sequence
    console.print("\n[dim]Shutting down gateway...[/dim]")

    tasks_to_cancel = [consumer_task, cron_task, heartbeat_task]
    if api_task is not None:
        tasks_to_cancel.append(api_task)

    for task in tasks_to_cancel:
        task.cancel()

    for task in tasks_to_cancel:
        with contextlib.suppress(asyncio.CancelledError):
            await task

    await cron_svc.stop()
    await heartbeat_svc.stop()
    await channel_mgr.stop_all()

    logger.info("Gateway shutdown complete")
    console.print("[green]Gateway stopped.[/green]")


async def _consume_inbound(
    bus: MessageBus,
    engine: EngineProtocol,
    session_mgr: SessionManager,
    memory_mgr: MemoryManager,
    config: GripConfig,
    trust_mgr: TrustManager,
) -> None:
    """Continuously pop inbound messages from the bus and process them.

    Handles control commands (/new, /clear, /compact, /undo, /model, /status, /trust)
    from channel bot handlers, and forwards normal messages to the engine.
    """
    # Per-session model overrides set via /model command
    session_models: dict[str, str] = {}

    while True:
        msg: InboundMessage = await bus.pop_inbound()
        session_key = f"{msg.channel}:{msg.chat_id}"

        # Handle control commands sent by channel bot handlers
        command = msg.metadata.get("command", "")

        if command == "new":
            session_mgr.delete(session_key)
            session_models.pop(session_key, None)
            logger.info("Session '{}' reset via /new command", session_key)
            continue

        if command == "clear":
            session = session_mgr.get_or_create(session_key)
            session.messages.clear()
            session.summary = None
            session_mgr.save(session)
            logger.info("Session '{}' cleared via /clear command", session_key)
            continue

        if command == "undo":
            session = session_mgr.get_or_create(session_key)
            if session.message_count < 2:
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text="Nothing to undo.",
                    )
                )
            else:
                session.messages = session.messages[:-2]
                session_mgr.save(session)
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text="Last exchange removed.",
                    )
                )
            continue

        if command == "compact":
            session = session_mgr.get_or_create(session_key)
            if session.message_count < 4:
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text="Session too short to compact.",
                    )
                )
                continue
            try:
                await engine.consolidate_session(session_key)
                # Re-fetch the session to get the updated message count
                session = session_mgr.get_or_create(session_key)
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=f"Session compacted. {session.message_count} messages remain.",
                    )
                )
            except Exception as exc:
                logger.error("Compact failed for {}: {}", session_key, exc)
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=f"Compact failed: {exc}",
                    )
                )
            continue

        if command == "model":
            model_name = msg.metadata.get("model_name", "").strip()
            if model_name:
                session_models[session_key] = model_name
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=f"Model switched to: {model_name}",
                    )
                )
            else:
                current = session_models.get(session_key, config.agents.defaults.model)
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=f"Current model: {current}",
                    )
                )
            continue

        if command == "status":
            session = session_mgr.get_or_create(session_key)
            model = session_models.get(session_key, config.agents.defaults.model)
            mem = memory_mgr.read_memory()
            mem_lines = len(mem.strip().splitlines()) if mem.strip() else 0
            trusted = trust_mgr.trusted_directories
            trust_line = ", ".join(trusted) if trusted else "none"
            status_text = (
                f"Session: {session_key}\n"
                f"Messages: {session.message_count}\n"
                f"Model: {model}\n"
                f"Memory facts: ~{mem_lines} lines\n"
                f"Trusted dirs: {trust_line}"
            )
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    text=status_text,
                )
            )
            continue

        if command == "trust":
            trust_path_str = msg.metadata.get("trust_path", "").strip()
            if trust_path_str:
                from pathlib import Path

                parts = trust_path_str.split(maxsplit=1)

                if parts[0] == "revoke" and len(parts) > 1:
                    revoke_path = Path(parts[1]).expanduser().resolve()
                    if trust_mgr.revoke(revoke_path):
                        text = f"Revoked trust for: {revoke_path}"
                    else:
                        text = f"Directory not found in trusted list: {revoke_path}"
                else:
                    trust_path = Path(trust_path_str).expanduser().resolve()
                    trust_mgr.trust(trust_path)
                    text = f"Trusted: {trust_path}"

                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=text,
                    )
                )
            else:
                dirs = trust_mgr.trusted_directories
                if dirs:
                    text = "Trusted directories:\n" + "\n".join(f"  {d}" for d in dirs)
                else:
                    text = "No directories trusted yet. Use /trust ~/path to trust a directory."
                await bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        text=text,
                    )
                )
            continue

        # Normal message -- run through the engine
        model_override = session_models.get(session_key)
        try:
            result = await engine.run(
                msg.text,
                session_key=session_key,
                model=model_override,
            )
            outbound = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                text=result.response,
                metadata={"iterations": result.iterations},
            )
            await bus.publish_outbound(outbound)
        except Exception as exc:
            logger.error(
                "Agent run failed for {}: {}",
                session_key,
                exc,
            )
            error_msg = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                text=f"Sorry, an error occurred: {exc}",
            )
            await bus.publish_outbound(error_msg)


def _wire_engine_messaging(engine: EngineProtocol, bus: MessageBus) -> None:
    """Connect outbound messaging to the engine based on its concrete type.

    LiteLLMRunner exposes a .registry with MessageTool/SendFileTool that need
    their callbacks wired to the bus. SDKRunner uses set_send_callback() and
    set_send_file_callback() to receive async callables directly.
    """
    from grip.engines.litellm_engine import LiteLLMRunner
    from grip.engines.sdk_engine import SDKRunner

    if isinstance(engine, LiteLLMRunner):
        _wire_message_tool(engine.registry, bus)
    elif isinstance(engine, SDKRunner):

        async def _send_via_bus(session_key: str, text: str) -> None:
            parts = session_key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "cli"
            chat_id = parts[1] if len(parts) > 1 else session_key
            await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, text=text))

        async def _send_file_via_bus(session_key: str, file_path: str, caption: str) -> None:
            parts = session_key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "cli"
            chat_id = parts[1] if len(parts) > 1 else session_key
            await bus.publish_outbound(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    text=caption,
                    file_path=file_path,
                )
            )

        engine.set_send_callback(_send_via_bus)
        engine.set_send_file_callback(_send_file_via_bus)


def _wire_message_tool(registry, bus: MessageBus) -> None:
    """Connect the send_message and send_file tools to the outbound message bus.

    Used by LiteLLMRunner which exposes a ToolRegistry with MessageTool and
    SendFileTool instances that accept async callbacks.
    """
    tool = registry.get("send_message")
    if isinstance(tool, MessageTool):

        async def _send_via_bus(session_key: str, text: str) -> None:
            parts = session_key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "cli"
            chat_id = parts[1] if len(parts) > 1 else session_key
            await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, text=text))

        tool.set_callback(_send_via_bus)

    file_tool = registry.get("send_file")
    if isinstance(file_tool, SendFileTool):

        async def _send_file_via_bus(session_key: str, file_path: str, caption: str) -> None:
            parts = session_key.split(":", 1)
            channel = parts[0] if len(parts) > 1 else "cli"
            chat_id = parts[1] if len(parts) > 1 else session_key
            await bus.publish_outbound(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    text=caption,
                    file_path=file_path,
                )
            )

        file_tool.set_callback(_send_file_via_bus)


def _start_api_server(
    config: GripConfig,
    engine: EngineProtocol,
    session_mgr: SessionManager,
    memory_mgr: MemoryManager,
    workspace: WorkspaceManager,
    cron_svc: CronService,
) -> asyncio.Task | None:
    """Start the REST API as a background asyncio task if dependencies are installed.

    Shares the same engine and managers with the gateway so channels and API
    operate on the same state. Sets app.state.engine for API routers, and
    app.state.tool_registry from the LiteLLMRunner registry (or None for SDKRunner).
    """
    from grip.api import is_available

    if not is_available():
        logger.debug("API dependencies not installed, skipping API server")
        return None

    import time

    import uvicorn
    from fastapi import FastAPI

    from grip.api.auth import ensure_auth_token
    from grip.api.errors import register_error_handlers
    from grip.api.middleware import (
        AuditLogMiddleware,
        RequestSizeLimitMiddleware,
        SecurityHeadersMiddleware,
    )
    from grip.api.rate_limit import SlidingWindowRateLimiter
    from grip.api.routers import chat, health, management, mcp, sessions, tools
    from grip.skills.loader import SkillsLoader

    api_config = config.gateway.api
    auth_token = ensure_auth_token(config, None)

    app = FastAPI(
        title="grip API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Populate app.state with shared gateway objects
    app.state.config = config
    app.state.auth_token = auth_token
    app.state.engine = engine
    # tool_registry is available on LiteLLMRunner; SDKRunner does not
    # expose one, so fall back to None
    app.state.tool_registry = getattr(engine, "registry", None)
    app.state.session_mgr = session_mgr
    app.state.memory_mgr = memory_mgr
    app.state.cron_service = cron_svc
    app.state.workspace = workspace
    app.state.start_time = time.time()

    skills_loader = SkillsLoader(workspace.root)
    skills_loader.scan()
    app.state.skills_loader = skills_loader

    app.state.ip_rate_limiter = SlidingWindowRateLimiter(
        max_requests=api_config.rate_limit_per_minute_per_ip,
        window_seconds=60,
    )
    app.state.token_rate_limiter = SlidingWindowRateLimiter(
        max_requests=api_config.rate_limit_per_minute,
        window_seconds=60,
    )

    # Middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=api_config.max_request_body_bytes)

    register_error_handlers(app)

    app.include_router(health.public_router)
    app.include_router(health.authed_router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(tools.router)
    app.include_router(management.router)
    app.include_router(mcp.router)

    uv_config = uvicorn.Config(
        app,
        host=config.gateway.host,
        port=config.gateway.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    console.print(
        f"[green]API server started:[/green] http://{config.gateway.host}:{config.gateway.port}"
    )

    return asyncio.create_task(server.serve(), name="api-server")


def _install_signal_handlers(shutdown_event: asyncio.Event) -> None:
    """Install SIGINT/SIGTERM handlers that trigger graceful shutdown."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)
