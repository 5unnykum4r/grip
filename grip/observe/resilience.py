"""Error recovery and resilience utilities for grip.

Provides crash-safe session auto-save via signal handlers, startup
config validation, and graceful degradation helpers.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class CrashRecovery:
    """Installs signal handlers that auto-save session state on crash.

    When SIGTERM or SIGINT is received, all registered save callbacks
    are invoked before the process exits. This protects in-flight
    conversations from being lost on unexpected termination.
    """

    def __init__(self) -> None:
        self._callbacks: list[tuple[str, Any]] = []
        self._installed = False

    def register_save_callback(self, name: str, callback: Any) -> None:
        """Register a callable that will be invoked on crash/shutdown."""
        self._callbacks.append((name, callback))

    def install(self) -> None:
        """Install signal handlers for SIGTERM and SIGINT."""
        if self._installed:
            return

        def _handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.warning(
                "Received {} — running {} save callbacks", sig_name, len(self._callbacks)
            )
            for name, callback in self._callbacks:
                try:
                    callback()
                    logger.debug("Crash save '{}' succeeded", name)
                except Exception as exc:
                    logger.error("Crash save '{}' failed: {}", name, exc)
            sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        self._installed = True
        logger.debug("Crash recovery handlers installed ({} callbacks)", len(self._callbacks))


def validate_config_on_startup(config) -> list[str]:
    """Run startup-time validation checks on the config.

    Returns a list of warning messages (empty = all good).
    These are non-fatal warnings, not hard errors.
    """
    warnings: list[str] = []

    defaults = config.agents.defaults
    ws_path = defaults.workspace.expanduser().resolve()

    if not ws_path.exists():
        warnings.append(f"Workspace directory does not exist: {ws_path}")

    if defaults.model == "openrouter/anthropic/claude-sonnet-4":
        # Check if API key is configured
        has_key = bool(config.providers.get("openrouter", None))
        if not has_key:
            import os

            if not os.environ.get("OPENROUTER_API_KEY"):
                warnings.append(
                    "Default model uses OpenRouter but no API key configured. "
                    "Run 'grip onboard' or set OPENROUTER_API_KEY."
                )

    if config.gateway.api.enable_tool_execute and not config.tools.restrict_to_workspace:
        warnings.append(
            "Tool execution is enabled over API AND workspace sandbox is disabled. "
            "This allows unrestricted file/shell access over HTTP."
        )

    if config.gateway.host == "0.0.0.0" and not config.gateway.api.auth_token.get_secret_value():
        warnings.append(
            "API bound to 0.0.0.0 without an auth token. "
            "A token will be auto-generated on first API start."
        )

    return warnings


def check_workspace_health(workspace_path: Path) -> dict[str, bool]:
    """Quick health check on workspace directory structure.

    Returns a dict of {component: is_healthy} for each expected
    subdirectory and template file.
    """
    checks: dict[str, bool] = {}
    expected_dirs = ["sessions", "memory", "skills", "cron"]
    for dirname in expected_dirs:
        checks[dirname] = (workspace_path / dirname).is_dir()

    checks["AGENT.md"] = (workspace_path / "AGENT.md").is_file()
    checks["config_exists"] = Path("~/.grip/config.json").expanduser().is_file()

    return checks
