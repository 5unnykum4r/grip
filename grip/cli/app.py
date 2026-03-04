"""Main CLI application: registers all subcommands and global options."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

from grip.logging import setup_logging

app = typer.Typer(
    name="grip",
    help="grip - Async-first agentic AI platform.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

_console = Console(stderr=True)


def _check_root() -> None:
    """Warn and require confirmation when running as root."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    _console.print(
        "\n[bold red]WARNING: grip is running as root.[/bold red]\n"
        "[yellow]The AI agent can execute shell commands and modify files.\n"
        "Running as root gives it unrestricted access to the entire system.\n"
        "This is strongly discouraged — use a non-root user instead.[/yellow]\n"
    )
    if sys.stdin.isatty() and not Confirm.ask("[bold]Continue as root?[/bold]", default=False, console=_console):
        _console.print("[dim]Exiting.[/dim]")
        sys.exit(1)


class _GlobalState:
    """Shared state set by the top-level callback, consumed by subcommands."""

    config_path: Path | None = None
    verbose: bool = False
    quiet: bool = False
    dry_run: bool = False


state = _GlobalState()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show DEBUG-level logs."),  # noqa: B008
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress logs below WARNING."),  # noqa: B008
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to config.json."),  # noqa: B008
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate execution without writing files or running commands."
    ),  # noqa: B008
) -> None:
    """grip - Async-first agentic AI platform."""
    _check_root()
    state.verbose = verbose
    state.quiet = quiet
    state.config_path = config
    state.dry_run = dry_run
    setup_logging(verbose=verbose, quiet=quiet)
    if dry_run:
        _console.print(
            "[yellow]DRY RUN MODE — no files will be written, no commands will execute[/yellow]"
        )


# Register subcommands — import at bottom to avoid circular deps
from grip.cli.agent_cmd import agent_command  # noqa: E402
from grip.cli.config_cmd import config_app  # noqa: E402
from grip.cli.cron_cmd import cron_app  # noqa: E402
from grip.cli.gateway_cmd import gateway_command  # noqa: E402
from grip.cli.mcp_cmd import mcp_app  # noqa: E402
from grip.cli.onboard import onboard_command  # noqa: E402
from grip.cli.serve_cmd import serve_command  # noqa: E402
from grip.cli.skills_cmd import skills_app  # noqa: E402
from grip.cli.status_cmd import status_command  # noqa: E402
from grip.cli.update_cmd import update_command  # noqa: E402
from grip.cli.workflow_cmd import workflow_app  # noqa: E402

app.command(name="onboard", help="Initialize grip: set up provider, API key, and workspace.")(
    onboard_command
)
app.command(name="agent", help="Chat with the AI agent (interactive or one-shot).")(agent_command)
app.command(name="gateway", help="Run the full platform: channels + cron + heartbeat.")(
    gateway_command
)
app.command(name="serve", help="Start the REST API server (standalone).")(serve_command)
app.command(name="status", help="Show system status.")(status_command)
app.command(name="update", help="Pull latest source and re-sync dependencies.")(update_command)
app.add_typer(config_app, name="config", help="View and modify configuration.")
app.add_typer(cron_app, name="cron", help="Manage scheduled cron jobs.")
app.add_typer(mcp_app, name="mcp", help="Manage MCP server configurations.")
app.add_typer(skills_app, name="skills", help="Manage agent skills.")
app.add_typer(workflow_app, name="workflow", help="Manage and run multi-agent workflows.")
