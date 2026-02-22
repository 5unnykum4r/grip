"""Management endpoints for the grip REST API.

Provides read-only access to system status, masked config, cron jobs,
skills, and memory. Write operations are limited to cron CRUD and
cron enable/disable toggles.

Deliberately NOT exposed: config mutation, skill installation, hooks
management — all too dangerous for remote HTTP access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from grip.api.auth import require_auth
from grip.api.dependencies import (
    check_rate_limit,
    check_token_rate_limit,
    get_config,
    get_memory_mgr,
)
from grip.config.schema import GripConfig
from grip.memory.manager import MemoryManager

router = APIRouter(prefix="/api/v1", tags=["management"])


# ── Status ──


@router.get(
    "/status",
    dependencies=[Depends(check_rate_limit)],
)
async def get_status(
    request: Request,
    token: str = Depends(require_auth),
    config: GripConfig = Depends(get_config),  # noqa: B008
) -> dict:
    """System status — same data as `grip status` CLI."""
    check_token_rate_limit(request, token)

    defaults = config.agents.defaults
    ws_path = defaults.workspace.expanduser().resolve()

    sessions_dir = ws_path / "sessions"
    session_count = 0
    if sessions_dir.exists():
        session_count = len(list(sessions_dir.glob("*.json")))

    channels_status = {}
    for name in ("telegram", "discord", "slack"):
        ch = getattr(config.channels, name, None)
        channels_status[name] = ch.enabled if ch else False

    return {
        "model": defaults.model,
        "max_tokens": defaults.max_tokens,
        "temperature": defaults.temperature,
        "max_tool_iterations": defaults.max_tool_iterations,
        "workspace": str(ws_path),
        "session_count": session_count,
        "sandbox_enabled": config.tools.restrict_to_workspace,
        "shell_timeout": config.tools.shell_timeout,
        "mcp_server_count": len(config.tools.mcp_servers),
        "channels": channels_status,
        "heartbeat_enabled": config.heartbeat.enabled,
        "heartbeat_interval_minutes": config.heartbeat.interval_minutes,
        "tool_execute_enabled": config.gateway.api.enable_tool_execute,
    }


# ── Config (masked) ──


def _mask_secrets(obj: Any) -> Any:
    """Recursively mask strings that look like API keys or tokens.

    Replicates the logic from grip/cli/config_cmd.py:_mask_secrets()
    so the API returns the same masked output as `grip config show`.
    """
    import re

    if isinstance(obj, str):
        if len(obj) > 8 and any(
            kw in obj.lower() for kw in ("sk-", "key-", "token", "secret", "grip_")
        ):
            return obj[:4] + "***" + obj[-4:]
        if len(obj) > 20 and re.match(r"^[A-Za-z0-9_\-]+$", obj):
            return obj[:4] + "***" + obj[-4:]
        return obj
    elif isinstance(obj, dict):
        return {k: _mask_secrets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_mask_secrets(item) for item in obj]
    return obj


def _stringify_paths(obj: dict) -> None:
    """Convert Path values to strings recursively."""
    for key, value in obj.items():
        if isinstance(value, Path):
            obj[key] = str(value)
        elif isinstance(value, dict):
            _stringify_paths(value)


@router.get(
    "/config",
    dependencies=[Depends(check_rate_limit)],
)
async def get_config_masked(
    request: Request,
    token: str = Depends(require_auth),
    config: GripConfig = Depends(get_config),  # noqa: B008
) -> dict:
    """Return the full config with all secrets masked."""
    check_token_rate_limit(request, token)

    data = config.model_dump(mode="json")
    _stringify_paths(data)
    masked = _mask_secrets(data)
    return {"config": masked}


# ── Cron ──


class CronJobCreateRequest(BaseModel):
    """Request body for creating a cron job."""

    name: str = Field(..., min_length=1, max_length=128)
    schedule: str = Field(..., min_length=5, max_length=128)
    prompt: str = Field(..., min_length=1, max_length=10000)
    reply_to: str = Field(default="", max_length=256)


@router.get(
    "/cron",
    dependencies=[Depends(check_rate_limit)],
)
async def list_cron_jobs(
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """List all configured cron jobs."""
    check_token_rate_limit(request, token)

    cron_svc = getattr(request.app.state, "cron_service", None)
    if cron_svc is None:
        return {"jobs": [], "count": 0}

    jobs = cron_svc.list_jobs()
    return {
        "jobs": [job.to_dict() for job in jobs],
        "count": len(jobs),
    }


@router.post(
    "/cron",
    dependencies=[Depends(check_rate_limit)],
    status_code=status.HTTP_201_CREATED,
)
async def create_cron_job(
    body: CronJobCreateRequest,
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Create a new cron job."""
    check_token_rate_limit(request, token)

    cron_svc = getattr(request.app.state, "cron_service", None)
    if cron_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cron service not available",
        )

    job = cron_svc.add_job(body.name, body.schedule, body.prompt, reply_to=body.reply_to)
    return {"created": True, "job": job.to_dict()}


@router.delete(
    "/cron/{job_id}",
    dependencies=[Depends(check_rate_limit)],
)
async def delete_cron_job(
    job_id: str,
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Delete a cron job by ID."""
    check_token_rate_limit(request, token)

    cron_svc = getattr(request.app.state, "cron_service", None)
    if cron_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cron service not available",
        )

    if not cron_svc.remove_job(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cron job not found",
        )
    return {"deleted": True, "job_id": job_id}


@router.post(
    "/cron/{job_id}/enable",
    dependencies=[Depends(check_rate_limit)],
)
async def enable_cron_job(
    job_id: str,
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Enable a cron job."""
    check_token_rate_limit(request, token)

    cron_svc = getattr(request.app.state, "cron_service", None)
    if cron_svc is None or not cron_svc.enable_job(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cron job not found",
        )
    return {"enabled": True, "job_id": job_id}


@router.post(
    "/cron/{job_id}/disable",
    dependencies=[Depends(check_rate_limit)],
)
async def disable_cron_job(
    job_id: str,
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Disable a cron job."""
    check_token_rate_limit(request, token)

    cron_svc = getattr(request.app.state, "cron_service", None)
    if cron_svc is None or not cron_svc.disable_job(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cron job not found",
        )
    return {"disabled": True, "job_id": job_id}


# ── Skills ──


@router.get(
    "/skills",
    dependencies=[Depends(check_rate_limit)],
)
async def list_skills(
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """List all loaded agent skills."""
    check_token_rate_limit(request, token)

    skills_loader = getattr(request.app.state, "skills_loader", None)
    if skills_loader is None:
        return {"skills": [], "count": 0}

    skills = skills_loader.list_skills()
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "always_loaded": s.always_loaded,
                "source": str(s.source_path),
            }
            for s in skills
        ],
        "count": len(skills),
    }


# ── Memory ──


@router.get(
    "/memory",
    dependencies=[Depends(check_rate_limit)],
)
async def get_memory(
    request: Request,
    token: str = Depends(require_auth),
    memory_mgr: MemoryManager = Depends(get_memory_mgr),  # noqa: B008
) -> dict:
    """Read the contents of MEMORY.md."""
    check_token_rate_limit(request, token)

    content = memory_mgr.read_memory()
    return {"content": content, "length": len(content)}


@router.get(
    "/memory/search",
    dependencies=[Depends(check_rate_limit)],
)
async def search_memory(
    q: str,
    request: Request,
    token: str = Depends(require_auth),
    memory_mgr: MemoryManager = Depends(get_memory_mgr),  # noqa: B008
) -> dict:
    """Search HISTORY.md for lines matching the query."""
    check_token_rate_limit(request, token)

    if not q or len(q) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be 1-500 characters",
        )

    results = memory_mgr.search_history(q)
    return {"query": q, "results": results, "count": len(results)}


# ── Metrics ──


@router.get(
    "/metrics",
    dependencies=[Depends(check_rate_limit)],
)
async def get_metrics(
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Return in-memory metrics snapshot."""
    check_token_rate_limit(request, token)

    from grip.observe.metrics import get_metrics as _get_metrics

    metrics = _get_metrics()
    return {"metrics": metrics.snapshot().to_dict()}


# ── Workflows ──


@router.get(
    "/workflows",
    dependencies=[Depends(check_rate_limit)],
)
async def list_workflows(
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """List all saved workflow definitions."""
    check_token_rate_limit(request, token)

    store = _get_workflow_store(request)
    if store is None:
        return {"workflows": [], "count": 0}

    names = store.list_workflows()
    workflows = []
    for name in names:
        wf = store.load(name)
        if wf:
            workflows.append(
                {
                    "name": wf.name,
                    "description": wf.description,
                    "step_count": len(wf.steps),
                }
            )
    return {"workflows": workflows, "count": len(workflows)}


@router.get(
    "/workflows/{name}",
    dependencies=[Depends(check_rate_limit)],
)
async def get_workflow(
    name: str,
    request: Request,
    token: str = Depends(require_auth),
) -> dict:
    """Get a workflow definition by name."""
    check_token_rate_limit(request, token)

    store = _get_workflow_store(request)
    if store is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")

    wf = store.load(name)
    if not wf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")

    return {"workflow": wf.to_dict()}


def _get_workflow_store(request: Request):
    """Get the workflow store from app.state, or create one from workspace."""
    store = getattr(request.app.state, "workflow_store", None)
    if store is not None:
        return store

    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        return None

    from grip.workflow.store import WorkflowStore

    return WorkflowStore(workspace.root / "workflows")
