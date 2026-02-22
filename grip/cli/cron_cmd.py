"""grip cron — manage scheduled tasks.

Subcommands:
  grip cron list         Show all cron jobs
  grip cron add          Add a new cron job
  grip cron remove       Remove a cron job by ID
  grip cron enable       Enable a disabled job
  grip cron disable      Disable a job without removing it
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from grip.config import load_config
from grip.cron.service import CronService

console = Console()
cron_app = typer.Typer(no_args_is_help=True)


def _get_cron_service() -> CronService:
    """Build a CronService from the current config (read-only, no engine)."""
    from grip.cli.app import state

    config = load_config(state.config_path)
    ws_path = config.agents.defaults.workspace.expanduser().resolve()
    # Pass None as engine -- CLI only manages jobs, doesn't execute them
    return CronService(ws_path / "cron", engine=None, config=config.cron)  # type: ignore[arg-type]


@cron_app.command(name="list")
def cron_list() -> None:
    """Show all scheduled cron jobs."""
    svc = _get_cron_service()
    jobs = svc.list_jobs()

    if not jobs:
        console.print("[dim]No cron jobs configured.[/dim]")
        return

    table = Table(title="Cron Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule", style="green")
    table.add_column("Enabled")
    table.add_column("Last Run", style="dim")
    table.add_column("Prompt", max_width=40)

    for job in jobs:
        enabled = "[green]Yes[/green]" if job.enabled else "[red]No[/red]"
        last_run = job.last_run[:19] if job.last_run else "Never"
        table.add_row(job.id, job.name, job.schedule, enabled, last_run, job.prompt[:40])

    console.print(table)


@cron_app.command(name="add")
def cron_add(
    name: str = typer.Argument(help="Job name."),  # noqa: B008
    schedule: str = typer.Argument(help="Cron expression (e.g. '*/5 * * * *')."),  # noqa: B008
    prompt: str = typer.Argument(help="Prompt to send to the agent when the job fires."),  # noqa: B008
    reply_to: str = typer.Option(
        "", "--reply-to", "-r", help="Session key to route results to (e.g. 'telegram:12345')."
    ),  # noqa: B008
) -> None:
    """Add a new cron job."""
    svc = _get_cron_service()
    job = svc.add_job(name, schedule, prompt, reply_to=reply_to)
    console.print(f"[green]Job added:[/green] {job.id} ({job.name})")
    console.print(f"  Schedule: {job.schedule}")
    if job.reply_to:
        console.print(f"  Reply to: {job.reply_to}")
    console.print(f"  Prompt: {job.prompt[:80]}")


@cron_app.command(name="remove")
def cron_remove(
    job_id: str = typer.Argument(help="Job ID to remove."),  # noqa: B008
) -> None:
    """Remove a cron job by ID."""
    svc = _get_cron_service()
    if svc.remove_job(job_id):
        console.print(f"[green]Removed:[/green] {job_id}")
    else:
        console.print(f"[red]Job not found:[/red] {job_id}")
        raise typer.Exit(1)


@cron_app.command(name="enable")
def cron_enable(
    job_id: str = typer.Argument(help="Job ID to enable."),  # noqa: B008
) -> None:
    """Enable a disabled cron job."""
    svc = _get_cron_service()
    if svc.enable_job(job_id):
        console.print(f"[green]Enabled:[/green] {job_id}")
    else:
        console.print(f"[red]Job not found:[/red] {job_id}")
        raise typer.Exit(1)


@cron_app.command(name="disable")
def cron_disable(
    job_id: str = typer.Argument(help="Job ID to disable."),  # noqa: B008
) -> None:
    """Disable a cron job without removing it."""
    svc = _get_cron_service()
    if svc.disable_job(job_id):
        console.print(f"[yellow]Disabled:[/yellow] {job_id}")
    else:
        console.print(f"[red]Job not found:[/red] {job_id}")
        raise typer.Exit(1)
