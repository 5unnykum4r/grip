"""grip config — view and modify configuration.

Subcommands:
  grip config show   Print current config (API keys masked)
  grip config set    Update a config value by dot-path
  grip config path   Print the config file path
"""

from __future__ import annotations

import json
import re
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from grip.config import GripConfig, get_config_path, load_config, save_config

console = Console()
config_app = typer.Typer(no_args_is_help=True)


def _mask_secrets(obj: Any) -> Any:
    """Recursively mask strings that look like API keys or tokens."""
    if isinstance(obj, str):
        if len(obj) > 8 and any(kw in obj.lower() for kw in ("sk-", "key-", "token", "secret")):
            return obj[:4] + "***" + obj[-4:]
        if len(obj) > 20 and re.match(r"^[A-Za-z0-9_\-]+$", obj):
            return obj[:4] + "***" + obj[-4:]
        return obj
    elif isinstance(obj, dict):
        return {
            k: _mask_secrets(v) if _is_secret_key(k) else _mask_secrets(v) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_mask_secrets(item) for item in obj]
    return obj


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return any(kw in key_lower for kw in ("api_key", "token", "secret", "password"))


@config_app.command(name="show")
def config_show() -> None:
    """Print current configuration (API keys masked)."""
    from grip.cli.app import state

    config = load_config(state.config_path)
    data = config.model_dump(mode="json")

    # Convert Path objects to strings
    _stringify_paths(data)

    masked = _mask_secrets(data)
    formatted = json.dumps(masked, indent=2, ensure_ascii=False)

    console.print(
        Panel(
            Syntax(formatted, "json", theme="monokai"),
            title="[bold cyan]grip Config[/bold cyan]",
            subtitle=f"[dim]{get_config_path()}[/dim]",
            expand=False,
        )
    )


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(help="Dot-separated config path (e.g. agents.defaults.model)."),  # noqa: B008
    value: str = typer.Argument(help="New value to set."),  # noqa: B008
) -> None:
    """Update a configuration value by dot-path."""
    from grip.cli.app import state

    config = load_config(state.config_path)
    data = config.model_dump(mode="json")
    _stringify_paths(data)

    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            console.print(f"[red]Error: Invalid config path '{key}'. '{part}' not found.[/red]")
            raise typer.Exit(1)

    final_key = parts[-1]
    if isinstance(target, dict) and final_key in target:
        old_value = target[final_key]
        target[final_key] = _coerce_value(value, old_value)
    else:
        console.print(f"[red]Error: Invalid config path '{key}'. '{final_key}' not found.[/red]")
        raise typer.Exit(1)

    try:
        updated_config = GripConfig(**data)
    except Exception as exc:
        console.print(f"[red]Validation error: {exc}[/red]")
        raise typer.Exit(1) from exc

    save_config(updated_config, state.config_path)
    console.print(f"[green]Updated[/green] {key} = {value}")


@config_app.command(name="path")
def config_path() -> None:
    """Print the config file path."""
    console.print(str(get_config_path()))


def _coerce_value(new: str, old: Any) -> Any:
    """Coerce a string value to match the type of the existing value."""
    if isinstance(old, bool):
        return new.lower() in ("true", "1", "yes")
    if isinstance(old, int):
        return int(new)
    if isinstance(old, float):
        return float(new)
    return new


def _stringify_paths(obj: dict) -> None:
    """Recursively convert Path-like values to strings."""
    from pathlib import Path

    for key, value in obj.items():
        if isinstance(value, Path):
            obj[key] = str(value)
        elif isinstance(value, dict):
            _stringify_paths(value)
