"""grip agent — one-shot and interactive chat with the AI agent.

One-shot:   grip agent -m "What time is it?"
Interactive: grip agent
Piped:      cat error.log | grip agent -m "Fix this"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

from grip.config import GripConfig, load_config
from grip.engines.factory import create_engine
from grip.engines.types import AgentRunResult, EngineProtocol
from grip.memory import MemoryManager
from grip.session import SessionManager
from grip.workspace import WorkspaceManager

console = Console()

_SESSION_KEY = "cli:interactive"


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

    # Read piped stdin (e.g. `cat error.log | grip agent -m "fix this"`)
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

    # Seed TOOLS.md at startup so it exists before the first run().
    # The LiteLLM engine regenerates it on every call to pick up new skills;
    # the SDK engine does not use TOOLS.md, but generating it here is harmless.
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
    """Run an interactive chat session with prompt-toolkit input."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    engine, session_mgr, memory_mgr = _build_engine(config)
    session_key = _SESSION_KEY

    history_dir = Path("~/.grip/history").expanduser()
    history_dir.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_dir / "cli_history")),
    )

    console.print(
        Panel(
            "[bold cyan]grip Interactive Mode[/bold cyan]\n"
            "Type your message and press Enter. Type [cyan]/help[/cyan] for commands.",
            expand=False,
        )
    )

    while True:
        try:
            user_input = await asyncio.to_thread(prompt_session.prompt, "\ngrip> ")
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
                console.print("[green]New session started.[/green]")
                continue

            elif cmd == "/clear":
                session = session_mgr.get_or_create(session_key)
                count = session.message_count
                session.messages.clear()
                session.summary = None
                session_mgr.save(session)
                console.print(f"[green]Cleared {count} messages from session.[/green]")
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
                with Live(
                    Spinner("dots", text="Compacting session..."),
                    console=console,
                    transient=True,
                ):
                    await engine.consolidate_session(session_key)
                # Re-read the session to get updated message count after compaction
                session = session_mgr.get_or_create(session_key)
                console.print(
                    f"[green]Session compacted. {session.message_count} messages remain.[/green]"
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
                    console.print(f"[green]Model switched to: {model}[/green]")
                else:
                    active = model or config.agents.defaults.model
                    console.print(f"  Active model: [cyan]{active}[/cyan]")
                continue

            elif cmd == "/doctor":
                console.print("[bold]grip Doctor[/bold]")
                console.print("  [green]Config loaded[/green]")
                model_str = model or config.agents.defaults.model
                provider_name = model_str.split("/")[0] if "/" in model_str else ""
                provider_cfg = config.providers.get(provider_name)
                if provider_cfg and provider_cfg.api_key:
                    console.print(f"  [green]Provider '{provider_name}' API key set[/green]")
                elif provider_name:
                    console.print(
                        f"  [yellow]Provider '{provider_name}' API key not in config "
                        f"(may be set via env var)[/yellow]"
                    )
                ws_path = config.agents.defaults.workspace.expanduser().resolve()
                if (ws_path / "AGENT.md").exists():
                    console.print(f"  [green]Workspace initialized: {ws_path}[/green]")
                else:
                    console.print(f"  [yellow]Workspace missing AGENT.md: {ws_path}[/yellow]")
                mcp_count = len(config.tools.mcp_servers)
                console.print(f"  MCP servers configured: {mcp_count}")
                session = session_mgr.get_or_create(session_key)
                console.print(f"  Session messages: {session.message_count}")
                continue

            elif cmd == "/mcp":
                servers = config.tools.mcp_servers
                if not servers:
                    console.print("[dim]No MCP servers configured.[/dim]")
                    continue
                console.print("[bold]MCP Servers[/bold]")
                for name, srv in servers.items():
                    transport = srv.url if srv.url else f"{srv.command} {' '.join(srv.args)}"
                    console.print(f"  [cyan]{name}[/cyan]  {transport}")
                continue

            elif cmd == "/tasks":
                ws_path = config.agents.defaults.workspace.expanduser().resolve()
                cron_dir = ws_path / "cron"
                if not cron_dir.exists() or not any(cron_dir.iterdir()):
                    console.print("[dim]No scheduled tasks found.[/dim]")
                    continue
                console.print("[bold]Scheduled Tasks[/bold]")
                for f in sorted(cron_dir.iterdir()):
                    if f.is_file():
                        console.print(f"  [cyan]{f.stem}[/cyan]  {f.suffix}")
                continue

            elif cmd == "/status":
                session = session_mgr.get_or_create(session_key)
                console.print(f"  Session: [cyan]{session_key}[/cyan]")
                console.print(f"  Messages: {session.message_count}")
                console.print(f"  Model: [cyan]{model or config.agents.defaults.model}[/cyan]")
                mem = memory_mgr.read_memory()
                mem_lines = len(mem.strip().splitlines()) if mem.strip() else 0
                console.print(f"  Memory facts: ~{mem_lines} lines")
                continue

            elif cmd == "/help":
                console.print(
                    "  [cyan]/new[/cyan]        Start a fresh session\n"
                    "  [cyan]/clear[/cyan]      Clear all messages in current session\n"
                    "  [cyan]/undo[/cyan]       Remove last exchange\n"
                    "  [cyan]/rewind N[/cyan]   Remove last N exchanges\n"
                    "  [cyan]/compact[/cyan]    Summarize and compress session history\n"
                    "  [cyan]/copy[/cyan]       Copy last response to clipboard\n"
                    "  [cyan]/model[/cyan]      Show or switch model (/model NAME)\n"
                    "  [cyan]/doctor[/cyan]     Check config, provider, and workspace health\n"
                    "  [cyan]/mcp[/cyan]        List configured MCP servers\n"
                    "  [cyan]/tasks[/cyan]      List scheduled cron tasks\n"
                    "  [cyan]/status[/cyan]     Session info\n"
                    "  [cyan]/help[/cyan]       This message\n"
                    "  [cyan]/exit[/cyan]       Quit"
                )
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
