"""grip workflow — manage and execute multi-agent workflows.

Subcommands:
  grip workflow list          List saved workflow definitions
  grip workflow show <name>   Show workflow steps and dependencies
  grip workflow run <name>    Execute a workflow
  grip workflow create        Create a workflow from a JSON file
  grip workflow delete <name> Delete a saved workflow
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from grip.config import load_config

console = Console()
workflow_app = typer.Typer(no_args_is_help=True)


def _get_store():
    from grip.cli.app import state

    config = load_config(state.config_path)
    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    from grip.workflow.store import WorkflowStore

    return WorkflowStore(ws_path / "workflows"), config


@workflow_app.command(name="list")
def workflow_list() -> None:
    """List all saved workflow definitions."""
    store, _ = _get_store()
    names = store.list_workflows()
    if not names:
        console.print("[dim]No workflows saved. Create one with: grip workflow create <file>[/dim]")
        return

    table = Table(title="Workflows")
    table.add_column("Name", style="bold")
    table.add_column("Steps")
    table.add_column("Description")

    for name in names:
        wf = store.load(name)
        if wf:
            table.add_row(wf.name, str(len(wf.steps)), wf.description[:60])

    console.print(table)


@workflow_app.command(name="show")
def workflow_show(
    name: str = typer.Argument(help="Workflow name to display."),  # noqa: B008
) -> None:
    """Show workflow steps, dependencies, and execution order."""
    store, _ = _get_store()
    wf = store.load(name)
    if not wf:
        console.print(f"[red]Workflow '{name}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{wf.name}[/bold cyan]")
    if wf.description:
        console.print(f"[dim]{wf.description}[/dim]")

    errors = wf.validate()
    if errors:
        console.print(f"[red]Validation errors: {'; '.join(errors)}[/red]")
        return

    layers = wf.get_execution_order()
    for i, layer in enumerate(layers, 1):
        console.print(f"\n  [bold]Layer {i}[/bold] (parallel):")
        for step_name in layer:
            step = next(s for s in wf.steps if s.name == step_name)
            deps = f" ← [{', '.join(step.depends_on)}]" if step.depends_on else ""
            console.print(f"    [{step.profile}] {step.name}{deps}")
            console.print(
                f"      [dim]{step.prompt[:80]}...[/dim]"
                if len(step.prompt) > 80
                else f"      [dim]{step.prompt}[/dim]"
            )


@workflow_app.command(name="run")
def workflow_run(
    name: str = typer.Argument(help="Workflow name to execute."),  # noqa: B008
) -> None:
    """Execute a workflow through the multi-agent engine."""
    store, config = _get_store()
    wf = store.load(name)
    if not wf:
        console.print(f"[red]Workflow '{name}' not found.[/red]")
        raise typer.Exit(1)

    errors = wf.validate()
    if errors:
        console.print(f"[red]Cannot run: {'; '.join(errors)}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Running workflow:[/bold] {wf.name} ({len(wf.steps)} steps)")
    result = asyncio.run(_run_workflow(config, wf))

    console.print(f"\n[bold]Result:[/bold] {result.status} ({result.total_duration_seconds:.1f}s)")
    for step_name, step_result in result.step_results.items():
        icon = "✓" if step_result.status.value == "completed" else "✗"
        console.print(
            f"  {icon} {step_name}: {step_result.status.value} ({step_result.duration_seconds:.1f}s)"
        )
        if step_result.error:
            console.print(f"    [red]{step_result.error}[/red]")


async def _run_workflow(config, wf):
    from grip.agent.loop import AgentLoop
    from grip.memory.manager import MemoryManager
    from grip.providers.registry import create_provider
    from grip.session.manager import SessionManager
    from grip.tools import create_default_registry
    from grip.workflow.engine import WorkflowEngine
    from grip.workspace.manager import WorkspaceManager

    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    ws = WorkspaceManager(ws_path)
    if not ws.is_initialized:
        ws.initialize()

    provider = create_provider(config)
    registry = create_default_registry()
    session_mgr = SessionManager(ws.root / "sessions")
    memory_mgr = MemoryManager(ws.root)

    loop = AgentLoop(
        config,
        provider,
        ws,
        tool_registry=registry,
        session_manager=session_mgr,
        memory_manager=memory_mgr,
    )

    engine = WorkflowEngine(config, loop, registry)
    return await engine.run(wf)


@workflow_app.command(name="create")
def workflow_create(
    file: Path = typer.Argument(help="Path to workflow JSON file."),  # noqa: B008
) -> None:
    """Create a workflow from a JSON definition file."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        from grip.workflow.models import WorkflowDef

        wf = WorkflowDef.from_dict(data)
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"[red]Invalid workflow file: {exc}[/red]")
        raise typer.Exit(1) from exc

    errors = wf.validate()
    if errors:
        console.print(f"[red]Validation errors: {'; '.join(errors)}[/red]")
        raise typer.Exit(1)

    store, _ = _get_store()
    path = store.save(wf)
    console.print(f"[green]Workflow '{wf.name}' saved[/green] ({len(wf.steps)} steps) → {path}")


@workflow_app.command(name="delete")
def workflow_delete(
    name: str = typer.Argument(help="Workflow name to delete."),  # noqa: B008
) -> None:
    """Delete a saved workflow."""
    store, _ = _get_store()
    if store.delete(name):
        console.print(f"[green]Deleted workflow: {name}[/green]")
    else:
        console.print(f"[red]Workflow '{name}' not found.[/red]")
        raise typer.Exit(1)
