"""Config file I/O: load from JSON, merge env vars, save back to disk."""

from __future__ import annotations

import json
from pathlib import Path

from grip.config.schema import GripConfig

_DEFAULT_CONFIG_DIR = Path.home() / ".grip"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.json"


def get_config_path() -> Path:
    return _DEFAULT_CONFIG_FILE


def get_workspace_path(config: GripConfig | None = None) -> Path:
    if config is None:
        return Path.home() / ".grip" / "workspace"
    return config.agents.defaults.workspace.expanduser().resolve()


def load_config(path: Path | None = None) -> GripConfig:
    """Load config from JSON file, falling back to defaults if file is missing.

    Environment variables with GRIP_ prefix override file values.
    Nested keys use __ as delimiter (e.g. GRIP_AGENTS__DEFAULTS__MODEL).
    """
    config_path = path or _DEFAULT_CONFIG_FILE
    config_path = config_path.expanduser().resolve()

    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return GripConfig(**raw)

    return GripConfig()


def save_config(config: GripConfig, path: Path | None = None) -> Path:
    """Serialize current config to JSON and write to disk atomically.

    Uses temp-file-then-rename for crash safety.
    """
    config_path = path or _DEFAULT_CONFIG_FILE
    config_path = config_path.expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json")
    # Convert Path objects to strings for JSON serialization
    _stringify_paths(data)
    _strip_empty_providers(data)

    tmp_path = config_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.rename(config_path)
    return config_path


def _strip_empty_providers(data: dict) -> None:
    """Remove provider entries where api_key and default_model are both empty.

    Prevents unconfigured providers (like lmstudio with default values)
    from polluting config.json when the user never set them up.
    """
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    empty_keys = [
        name
        for name, entry in providers.items()
        if isinstance(entry, dict) and not entry.get("api_key") and not entry.get("default_model")
    ]
    for key in empty_keys:
        del providers[key]


def _stringify_paths(obj: dict) -> None:
    """Recursively convert any remaining Path-like values to strings."""
    for key, value in obj.items():
        if isinstance(value, Path):
            obj[key] = str(value)
        elif isinstance(value, dict):
            _stringify_paths(value)
