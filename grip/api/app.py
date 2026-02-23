"""FastAPI application factory for the grip REST API.

create_api_app() builds a fully wired FastAPI instance with:
  - Lifespan that initializes the full agent stack (provider, tools, sessions, memory)
  - Security middleware (size limit, audit log, security headers, optional CORS)
  - Sanitized error handlers
  - All route modules mounted

The app stores shared state (engine, registries, managers) on app.state
so that FastAPI dependency injection can retrieve them in route handlers.
"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from grip import __version__
from grip.api.auth import ensure_auth_token
from grip.api.errors import register_error_handlers
from grip.api.middleware import (
    AuditLogMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from grip.api.rate_limit import SlidingWindowRateLimiter
from grip.api.routers import chat, health, management, mcp, sessions, tools
from grip.config.schema import GripConfig
from grip.engines.factory import create_engine
from grip.memory.manager import MemoryManager
from grip.session.manager import SessionManager
from grip.skills.loader import SkillsLoader
from grip.workspace.manager import WorkspaceManager


def create_api_app(config: GripConfig, config_path: Path | None = None) -> FastAPI:
    """Build a fully wired FastAPI application.

    The returned app is ready to be passed to uvicorn.run(). All
    agent-related state is initialized in the lifespan context manager
    and stored on app.state for dependency injection.
    """
    api_config = config.gateway.api

    auth_token = ensure_auth_token(config, config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        ws_path = config.agents.defaults.workspace.expanduser().resolve()
        ws = WorkspaceManager(ws_path)
        if not ws.is_initialized:
            ws.initialize()

        session_mgr = SessionManager(ws.root / "sessions")
        memory_mgr = MemoryManager(ws.root)
        skills_loader = SkillsLoader(ws.root)
        skills_loader.scan()

        engine = create_engine(config, ws, session_mgr, memory_mgr)

        from grip.channels.direct import wire_direct_sender

        direct_sender = wire_direct_sender(engine, config.channels)

        app.state.config = config
        app.state.config_path = config_path
        app.state.auth_token = auth_token
        app.state.engine = engine
        app.state.tool_registry = getattr(engine, "registry", None)
        app.state.session_mgr = session_mgr
        app.state.memory_mgr = memory_mgr
        app.state.skills_loader = skills_loader
        app.state.workspace = ws
        app.state.start_time = time.time()
        app.state.ip_rate_limiter = SlidingWindowRateLimiter(
            max_requests=api_config.rate_limit_per_minute_per_ip,
            window_seconds=60,
        )
        app.state.token_rate_limiter = SlidingWindowRateLimiter(
            max_requests=api_config.rate_limit_per_minute,
            window_seconds=60,
        )

        _print_startup_warnings(config)
        logger.info("API server started on {}:{}", config.gateway.host, config.gateway.port)

        yield

        if direct_sender:
            await direct_sender.close()

        from grip.pool import shutdown_pools

        await shutdown_pools()
        logger.info("API server shutting down")

    app = FastAPI(
        title="grip API",
        version=__version__,
        description="Async-first agentic AI platform REST API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    # Middleware (outermost applied first = added last in FastAPI)
    if api_config.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors_allowed_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=api_config.max_request_body_bytes)

    register_error_handlers(app)

    # Mount routers
    app.include_router(health.public_router)
    app.include_router(health.authed_router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(tools.router)
    app.include_router(management.router)
    app.include_router(mcp.router)

    return app


def _print_startup_warnings(config: GripConfig) -> None:
    """Print security-relevant warnings to stderr on API startup."""
    host = config.gateway.host
    api_config = config.gateway.api

    if host == "0.0.0.0":
        print(
            "\n  WARNING: API bound to 0.0.0.0 — accessible from all network interfaces.\n"
            "  Use a reverse proxy (nginx/caddy) with HTTPS for production.\n",
            file=sys.stderr,
        )

    if host != "127.0.0.1":
        print(
            "  Note: API runs over HTTP only. Use a reverse proxy for HTTPS/TLS.\n",
            file=sys.stderr,
        )

    if api_config.enable_tool_execute:
        print(
            "  WARNING: Direct tool execution is ENABLED. This allows arbitrary\n"
            "  tool invocation (including shell) over HTTP.\n",
            file=sys.stderr,
        )

    if not config.tools.restrict_to_workspace:
        print(
            "  WARNING: Workspace sandbox is DISABLED. File tools can access\n"
            "  any path on the host filesystem.\n",
            file=sys.stderr,
        )
