"""grip agent — one-shot and interactive chat with the AI agent.

One-shot:   grip agent -m "What time is it?"
Interactive: grip agent
Piped:      cat error.log | grip agent -m "Fix this"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Iterable

import typer
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from grip.config import GripConfig, load_config
from grip.engines.factory import create_engine
from grip.engines.types import AgentRunResult, EngineProtocol
from grip.memory import MemoryManager
from grip.providers.registry import ProviderRegistry, PROVIDERS
from grip.session import SessionManager
from grip.workspace import WorkspaceManager

console = Console()

_SESSION_KEY = "cli:interactive"

# All slash commands with metadata for autocomplete and /help display.
# Structure: {command: (description, category)}
_COMMANDS: dict[str, tuple[str, str]] = {
    "/new": ("Start a fresh session and clear terminal", "Session"),
    "/clear": ("Clear all messages and terminal", "Session"),
    "/undo": ("Remove last exchange", "Session"),
    "/rewind": ("Remove last N exchanges (/rewind N)", "Session"),
    "/compact": ("Summarize, compress history, and clear terminal", "Session"),
    "/copy": ("Copy last response to clipboard", "Session"),
    "/model": ("Show or switch model (/model [provider/]name)", "Config"),
    "/provider": ("Show current provider details", "Config"),
    "/doctor": ("Check config, provider, and workspace health", "Info"),
    "/mcp": ("List configured MCP servers", "Info"),
    "/tasks": ("List scheduled cron tasks", "Info"),
    "/status": ("Show session and system info", "Info"),
    "/help": ("Show this command reference", "Info"),
    "/exit": ("Quit interactive mode", ""),
}


def _build_completer() -> Any:
    """Create a prompt_toolkit Completer that autocompletes slash commands.

    When the user types `/` or `/c`, matching commands with descriptions
    are shown in a dropdown. Regular text input is not affected.
    """
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document

    class SlashCompleter(Completer):
        """Autocomplete slash commands with descriptions in the dropdown."""

        def get_completions(
            self, document: Document, complete_event: Any
        ) -> Iterable[Completion]:
            text = document.text_before_cursor.lstrip()

            if not text.startswith("/"):
                return

            for cmd, (desc, _cat) in _COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )

    return SlashCompleter()


def _short_model_name(model_string: str) -> str:
    """Extract a display-friendly short name from a full model string.

    'openrouter/anthropic/claude-sonnet-4' -> 'claude-sonnet-4'
    'anthropic/claude-sonnet-4' -> 'claude-sonnet-4'
    'gpt-4o' -> 'gpt-4o'
    """
    parts = model_string.split("/")
    return parts[-1] if parts else model_string


def _resolve_provider_display(model_string: str, config: GripConfig) -> tuple[str, str]:
    """Resolve a model string to (provider_display_name, bare_model_name)."""
    explicit_provider = config.agents.defaults.provider
    try:
        spec, bare = ProviderRegistry.resolve_model(model_string, provider=explicit_provider)
        return spec.display_name, bare
    except ValueError:
        return "Unknown", model_string


def _read_stdin() -> str:
    """Read piped stdin if available (non-TTY). Returns empty string if no piped input."""
    if sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def agent_command(
    message: str | None = typer.Option(  # noqa: B008
        None, "--message", "-m", help="One-shot message. Omit for interactive mode."
    ),
    model: str | None = typer.Option(None, "--model", help="Override the default model."),  # noqa: B008
    no_markdown: bool = typer.Option(False, "--no-markdown", help="Plain text output."),  # noqa: B008
) -> None:
    """Chat with the AI agent."""
    from grip.cli.app import state

    config = load_config(state.config_path)
    if state.dry_run:
        config.agents.defaults.dry_run = True
    _ensure_workspace(config)

    piped_input = _read_stdin()
    if piped_input:
        piped_block = f"<stdin>\n{piped_input.strip()}\n</stdin>"
        message = f"{message}\n\n{piped_block}" if message else piped_block

    if message:
        asyncio.run(_one_shot(config, message, model=model, no_markdown=no_markdown))
    else:
        asyncio.run(_interactive(config, model=model, no_markdown=no_markdown))


def _ensure_workspace(config: GripConfig) -> WorkspaceManager:
    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    ws = WorkspaceManager(ws_path)
    if not ws.is_initialized:
        ws.initialize()
        console.print(f"[dim]Workspace initialized: {ws_path}[/dim]")
    return ws


def _build_engine(config: GripConfig) -> tuple[EngineProtocol, SessionManager, MemoryManager]:
    """Wire up the engine stack from config using the engine factory."""
    ws = _ensure_workspace(config)
    session_mgr = SessionManager(ws.root / "sessions")
    memory_mgr = MemoryManager(ws.root)

    from grip.skills.loader import SkillsLoader
    from grip.tools import create_default_registry
    from grip.tools.docs import generate_tools_md

    mcp_servers = config.tools.mcp_servers
    registry = create_default_registry(mcp_servers=mcp_servers)
    loader = SkillsLoader(ws.root)
    loader.scan()
    tools_md = generate_tools_md(registry, loader.list_skills(), config.tools.mcp_servers)
    (ws.root / "TOOLS.md").write_text(tools_md, encoding="utf-8")

    engine = create_engine(config, ws, session_mgr, memory_mgr)
    return engine, session_mgr, memory_mgr


def _print_response(text: str, no_markdown: bool) -> None:
    if no_markdown:
        console.print(text)
    else:
        console.print(Markdown(text))


def _print_stats(result: AgentRunResult) -> None:
    """Print execution stats. Shows a Rich table when tools were used."""
    if result.tool_details:
        table = Table(
            show_header=True,
            header_style="bold dim",
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Time", justify="right", style="dim", no_wrap=True)

        for td in result.tool_details:
            status = "[green]OK[/green]" if td.success else "[red]FAIL[/red]"
            if td.duration_ms >= 1000:
                time_str = f"{td.duration_ms / 1000:.1f}s"
            else:
                time_str = f"{td.duration_ms:.0f}ms"
            table.add_row(td.name, status, time_str)

        console.print(table)

    parts: list[str] = []
    if result.iterations > 1:
        parts.append(f"{result.iterations} iterations")
    if result.total_tokens > 0:
        parts.append(f"{result.total_tokens} tokens")
    if parts:
        console.print(f"[dim]({' | '.join(parts)})[/dim]")


def _print_welcome(config: GripConfig, model: str | None) -> None:
    """Print an enhanced welcome panel with model, provider, and engine info."""
    from grip import __version__

    active_model = model or config.agents.defaults.model
    provider_name, bare_model = _resolve_provider_display(active_model, config)
    engine_type = config.agents.defaults.engine

    info_lines = [
        f"[bold cyan]grip Interactive Mode[/bold cyan]  [dim]v{__version__}[/dim]",
        "",
        f"  [dim]Provider :[/dim]  [white]{provider_name}[/white]",
        f"  [dim]Model    :[/dim]  [cyan]{bare_model}[/cyan]",
        f"  [dim]Engine   :[/dim]  [white]{engine_type}[/white]",
        "",
        "  Type a message or [cyan]/help[/cyan] for commands.",
        "  Start typing [cyan]/[/cyan] to autocomplete commands.",
    ]
    console.print(Panel("\n".join(info_lines), expand=False, border_style="cyan"))


def _print_help() -> None:
    """Print categorized command help using a Rich Table."""
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    categories_order = ["Session", "Config", "Info"]
    for category in categories_order:
        table.add_row(f"[bold dim]{category}[/bold dim]", "")
        for cmd, (desc, cat) in _COMMANDS.items():
            if cat == category:
                table.add_row(f"  {cmd}", desc)

    # Add /exit at the bottom outside categories
    table.add_row("")
    table.add_row("  [cyan]/exit[/cyan]", "Quit interactive mode")

    console.print(Panel(table, title="[bold]Commands[/bold]", expand=False, border_style="dim"))


def _print_status(
    session_key: str,
    session_mgr: SessionManager,
    memory_mgr: MemoryManager,
    model: str | None,
    config: GripConfig,
) -> None:
    """Print an enhanced status panel with session, model, and memory info."""
    session = session_mgr.get_or_create(session_key)
    active_model = model or config.agents.defaults.model
    provider_name, bare_model = _resolve_provider_display(active_model, config)
    engine_type = config.agents.defaults.engine

    mem = memory_mgr.read_memory()
    mem_lines = len(mem.strip().splitlines()) if mem.strip() else 0

    has_summary = "Yes" if session.summary else "No"

    info_lines = [
        f"  [dim]Session  :[/dim]  [white]{session_key}[/white]",
        f"  [dim]Messages :[/dim]  [white]{session.message_count}[/white]",
        f"  [dim]Summary  :[/dim]  [white]{has_summary}[/white]",
        "",
        f"  [dim]Provider :[/dim]  [white]{provider_name}[/white]",
        f"  [dim]Model    :[/dim]  [cyan]{bare_model}[/cyan]",
        f"  [dim]Engine   :[/dim]  [white]{engine_type}[/white]",
        "",
        f"  [dim]Memory   :[/dim]  [white]{mem_lines} facts[/white]",
    ]
    console.print(Panel("\n".join(info_lines), title="[bold]Status[/bold]", expand=False, border_style="dim"))


def _print_doctor(
    session_key: str,
    session_mgr: SessionManager,
    model: str | None,
    config: GripConfig,
) -> None:
    """Run diagnostic checks and display results with pass/fail indicators."""
    checks: list[tuple[str, bool, str]] = []

    # Config check
    checks.append(("Config loaded", True, ""))

    # Provider / API key check
    active_model = model or config.agents.defaults.model
    provider_name, bare_model = _resolve_provider_display(active_model, config)

    provider_key_name = active_model.split("/")[0] if "/" in active_model else ""
    provider_cfg = config.providers.get(provider_key_name)
    if provider_cfg and provider_cfg.api_key:
        checks.append((f"Provider '{provider_name}' API key", True, ""))
    elif provider_key_name:
        checks.append((f"Provider '{provider_name}' API key", False, "May be set via env var"))
    else:
        checks.append((f"Provider '{provider_name}'", True, "No key required"))

    # Workspace check
    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    ws_ok = (ws_path / "AGENT.md").exists()
    checks.append(("Workspace initialized", ws_ok, str(ws_path)))

    # MCP servers
    mcp_count = len(config.tools.mcp_servers)
    checks.append((f"MCP servers ({mcp_count})", mcp_count > 0 or True, ""))

    # Session state
    session = session_mgr.get_or_create(session_key)
    checks.append((f"Session ({session.message_count} messages)", True, session_key))

    lines: list[str] = []
    for label, passed, detail in checks:
        icon = "[green]PASS[/green]" if passed else "[yellow]WARN[/yellow]"
        line = f"  {icon}  {label}"
        if detail:
            line += f"  [dim]({detail})[/dim]"
        lines.append(line)

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]grip Doctor[/bold]",
            expand=False,
            border_style="dim",
        )
    )


def _print_model_info(model: str | None, config: GripConfig) -> None:
    """Print detailed model and provider information."""
    active_model = model or config.agents.defaults.model
    provider_name, bare_model = _resolve_provider_display(active_model, config)

    lines = [
        f"  [dim]Provider   :[/dim]  [white]{provider_name}[/white]",
        f"  [dim]Model      :[/dim]  [cyan]{bare_model}[/cyan]",
        f"  [dim]Full ID    :[/dim]  [dim]{active_model}[/dim]",
    ]

    # Show other available providers for switching
    provider_names = [s.display_name for s in PROVIDERS[:9]]
    lines.append("")
    lines.append(f"  [dim]Switch with:[/dim]  /model [cyan]provider/model-name[/cyan]")
    lines.append(f"  [dim]Providers  :[/dim]  {', '.join(provider_names[:5])}, ...")

    console.print(
        Panel("\n".join(lines), title="[bold]Model[/bold]", expand=False, border_style="dim")
    )


async def _one_shot(
    config: GripConfig, message: str, *, model: str | None, no_markdown: bool
) -> None:
    """Send a single message, print the response, and exit."""
    engine, _, _ = _build_engine(config)

    with Live(Spinner("dots", text="Thinking..."), console=console, transient=True):
        result = await engine.run(message, session_key="cli:oneshot", model=model)

    _print_response(result.response, no_markdown)
    _print_stats(result)


async def _interactive(config: GripConfig, *, model: str | None, no_markdown: bool) -> None:
    """Run an interactive chat session with slash-command autocomplete and enhanced UI."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.formatted_text import HTML

    engine, session_mgr, memory_mgr = _build_engine(config)
    session_key = _SESSION_KEY

    history_dir = Path("~/.grip/history").expanduser()
    history_dir.mkdir(parents=True, exist_ok=True)

    completer = _build_completer()

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "cli_history")),
        completer=completer,
        complete_while_typing=True,
    )

    _print_welcome(config, model)

    def _make_prompt() -> HTML:
        """Build a dynamic prompt showing the active model's short name."""
        short = _short_model_name(model or config.agents.defaults.model)
        return HTML(f"\n<b>grip</b> <style fg='ansibrightcyan'>({short})</style><b>&gt;</b> ")

    while True:
        try:
            user_input = await asyncio.to_thread(prompt_session.prompt, _make_prompt())
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            args_parts = user_input.split()[1:]
            args_str = user_input[len(cmd) :].strip()

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/new":
                await engine.reset_session(session_key)
                session_key = _SESSION_KEY
                console.clear()
                _print_welcome(config, model)
                console.print("[green]New session started.[/green]")
                continue

            elif cmd == "/clear":
                session = session_mgr.get_or_create(session_key)
                count = session.message_count
                session.messages.clear()
                session.summary = None
                session_mgr.save(session)
                console.clear()
                _print_welcome(config, model)
                console.print(f"[green]Cleared {count} messages.[/green]")
                continue

            elif cmd == "/undo":
                session = session_mgr.get_or_create(session_key)
                if session.message_count < 2:
                    console.print("[yellow]Nothing to undo.[/yellow]")
                    continue
                session.messages = session.messages[:-2]
                session_mgr.save(session)
                console.print("[green]Last exchange removed.[/green]")
                continue

            elif cmd == "/rewind":
                try:
                    n = int(args_parts[0]) if args_parts else 1
                except ValueError:
                    console.print("[yellow]Usage: /rewind N (e.g. /rewind 3)[/yellow]")
                    continue
                session = session_mgr.get_or_create(session_key)
                to_remove = n * 2
                if session.message_count < to_remove:
                    console.print(
                        f"[yellow]Session only has {session.message_count // 2} "
                        f"exchange(s), cannot rewind {n}.[/yellow]"
                    )
                    continue
                session.messages = session.messages[:-to_remove]
                session_mgr.save(session)
                console.print(f"[green]Rewound {n} exchange(s).[/green]")
                continue

            elif cmd == "/compact":
                session = session_mgr.get_or_create(session_key)
                if session.message_count < 4:
                    console.print("[yellow]Session too short to compact.[/yellow]")
                    continue
                old_count = session.message_count
                with Live(
                    Spinner("dots", text="Compacting session..."),
                    console=console,
                    transient=True,
                ):
                    await engine.consolidate_session(session_key)

                session = session_mgr.get_or_create(session_key)
                console.clear()
                _print_welcome(config, model)
                console.print(
                    f"[green]Session compacted: {old_count} -> {session.message_count} messages.[/green]"
                )
                if session.summary:
                    console.print(
                        Panel(
                            session.summary,
                            title="[bold]Conversation Summary[/bold]",
                            expand=False,
                            border_style="dim",
                        )
                    )
                continue

            elif cmd == "/copy":
                session = session_mgr.get_or_create(session_key)
                last_assistant = next(
                    (m.content for m in reversed(session.messages) if m.role == "assistant"),
                    None,
                )
                if not last_assistant:
                    console.print("[yellow]No assistant response to copy.[/yellow]")
                    continue
                import subprocess as _sp

                os_name = config.platform.os
                try:
                    if os_name == "darwin":
                        _sp.run(["pbcopy"], input=last_assistant.encode(), check=True)
                    elif os_name == "windows":
                        _sp.run(["clip"], input=last_assistant.encode(), check=True)
                    else:
                        _sp.run(
                            ["xclip", "-selection", "clipboard"],
                            input=last_assistant.encode(),
                            check=True,
                        )
                    console.print("[green]Copied last response to clipboard.[/green]")
                except FileNotFoundError:
                    console.print(
                        "[red]Clipboard tool not found. "
                        "Install xclip (Linux) or use macOS/Windows.[/red]"
                    )
                except _sp.CalledProcessError as exc:
                    console.print(f"[red]Clipboard error: {exc}[/red]")
                continue

            elif cmd == "/model":
                if args_str:
                    model = args_str
                    provider_name, bare = _resolve_provider_display(model, config)
                    console.print(
                        f"[green]Model switched to [cyan]{bare}[/cyan] "
                        f"via [white]{provider_name}[/white][/green]"
                    )
                else:
                    _print_model_info(model, config)
                continue

            elif cmd == "/provider":
                _print_model_info(model, config)
                continue

            elif cmd == "/doctor":
                _print_doctor(session_key, session_mgr, model, config)
                continue

            elif cmd == "/mcp":
                servers = config.tools.mcp_servers
                if not servers:
                    console.print("[dim]No MCP servers configured.[/dim]")
                    continue
                lines = []
                for name, srv in servers.items():
                    transport = srv.url if srv.url else f"{srv.command} {' '.join(srv.args)}"
                    lines.append(f"  [cyan]{name}[/cyan]  {transport}")
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="[bold]MCP Servers[/bold]",
                        expand=False,
                        border_style="dim",
                    )
                )
                continue

            elif cmd == "/tasks":
                ws_path = config.agents.defaults.workspace.expanduser().resolve()
                cron_dir = ws_path / "cron"
                if not cron_dir.exists() or not any(cron_dir.iterdir()):
                    console.print("[dim]No scheduled tasks found.[/dim]")
                    continue
                lines = []
                for f in sorted(cron_dir.iterdir()):
                    if f.is_file():
                        lines.append(f"  [cyan]{f.stem}[/cyan]  {f.suffix}")
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="[bold]Scheduled Tasks[/bold]",
                        expand=False,
                        border_style="dim",
                    )
                )
                continue

            elif cmd == "/status":
                _print_status(session_key, session_mgr, memory_mgr, model, config)
                continue

            elif cmd == "/help":
                _print_help()
                continue

            else:
                console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")
                continue

        # Run the agent
        with Live(Spinner("dots", text="Thinking..."), console=console, transient=True):
            try:
                result = await engine.run(user_input, session_key=session_key, model=model)
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                logger.exception("Agent run failed")
                continue

        console.print()
        _print_response(result.response, no_markdown)
        _print_stats(result)
