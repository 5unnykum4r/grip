"""Tests for the TrustManager directory trust system."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from grip.trust import TrustManager


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Provide a temporary state directory."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def trust_mgr(state_dir: Path) -> TrustManager:
    return TrustManager(state_dir)


def test_workspace_always_trusted(trust_mgr: TrustManager, workspace: Path):
    """Paths inside the workspace are always trusted."""
    assert trust_mgr.is_trusted(workspace, workspace) is True
    assert trust_mgr.is_trusted(workspace / "subdir" / "file.txt", workspace) is True


def test_untrusted_path_denied(trust_mgr: TrustManager, workspace: Path, tmp_path: Path):
    """Paths outside workspace and not in trusted list are denied."""
    outside = tmp_path / "other" / "file.txt"
    assert trust_mgr.is_trusted(outside, workspace) is False


def test_trust_directory(trust_mgr: TrustManager, workspace: Path, tmp_path: Path):
    """Trusting a directory allows access to it and all subdirectories."""
    project = tmp_path / "projects"
    project.mkdir()

    assert trust_mgr.is_trusted(project / "file.txt", workspace) is False

    trust_mgr.trust(project)

    assert trust_mgr.is_trusted(project, workspace) is True
    assert trust_mgr.is_trusted(project / "file.txt", workspace) is True
    assert trust_mgr.is_trusted(project / "sub" / "deep" / "file.py", workspace) is True


def test_trust_persisted(state_dir: Path, workspace: Path, tmp_path: Path):
    """Trust decisions survive TrustManager recreation (persisted to JSON)."""
    mgr1 = TrustManager(state_dir)
    project = tmp_path / "myproject"
    project.mkdir()
    mgr1.trust(project)

    # Create a new instance from the same state directory
    mgr2 = TrustManager(state_dir)
    assert mgr2.is_trusted(project / "file.txt", workspace) is True


def test_revoke_trust(trust_mgr: TrustManager, workspace: Path, tmp_path: Path):
    """Revoking removes a directory from the trusted list."""
    project = tmp_path / "revokable"
    project.mkdir()

    trust_mgr.trust(project)
    assert trust_mgr.is_trusted(project / "file.txt", workspace) is True

    result = trust_mgr.revoke(project)
    assert result is True
    assert trust_mgr.is_trusted(project / "file.txt", workspace) is False


def test_revoke_nonexistent(trust_mgr: TrustManager, tmp_path: Path):
    """Revoking a directory that was never trusted returns False."""
    assert trust_mgr.revoke(tmp_path / "never_trusted") is False


def test_trusted_directories_property(trust_mgr: TrustManager, tmp_path: Path):
    """trusted_directories returns sorted list of trusted paths."""
    dir_a = tmp_path / "aaa"
    dir_b = tmp_path / "bbb"
    dir_a.mkdir()
    dir_b.mkdir()

    trust_mgr.trust(dir_b)
    trust_mgr.trust(dir_a)

    dirs = trust_mgr.trusted_directories
    assert dirs == [str(dir_a.resolve()), str(dir_b.resolve())]


def test_find_trust_target_home_subdir(tmp_path: Path, monkeypatch):
    """For paths under home, trust target is the first child of home."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    path = fake_home / "Downloads" / "project" / "file.txt"
    target = TrustManager.find_trust_target(path)
    assert target == fake_home / "Downloads"


def test_find_trust_target_outside_home(tmp_path: Path, monkeypatch):
    """For paths outside home, trust target is the first directory after root."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Use a path clearly outside fake_home
    outside = tmp_path / "external" / "project" / "file.txt"
    target = TrustManager.find_trust_target(outside)
    # Should trust the first component after root within the resolved path
    resolved = outside.resolve()
    expected = Path(resolved.root) / resolved.parts[1]
    assert target == expected


def test_check_and_prompt_trusted(trust_mgr: TrustManager, workspace: Path):
    """check_and_prompt returns True immediately for trusted paths."""
    result = asyncio.run(trust_mgr.check_and_prompt(workspace / "file.txt", workspace))
    assert result is True


def test_check_and_prompt_no_callback(trust_mgr: TrustManager, workspace: Path, tmp_path: Path):
    """Without a prompt callback, untrusted paths are denied."""
    outside = tmp_path / "outside" / "file.txt"
    result = asyncio.run(trust_mgr.check_and_prompt(outside, workspace))
    assert result is False


def test_check_and_prompt_granted(trust_mgr: TrustManager, workspace: Path, tmp_path: Path):
    """When the prompt callback returns True, the directory is trusted and persisted."""
    outside = tmp_path / "grantable"
    outside.mkdir()

    async def _always_grant(directory: Path) -> bool:
        return True

    trust_mgr.set_prompt(_always_grant)
    result = asyncio.run(trust_mgr.check_and_prompt(outside / "file.txt", workspace))
    assert result is True
    # Should now be persistently trusted
    assert trust_mgr.is_trusted(outside / "other.txt", workspace) is True


def test_check_and_prompt_denied_caches_session(
    trust_mgr: TrustManager, workspace: Path, tmp_path: Path
):
    """When denied, the same directory is not re-prompted in the same session."""
    outside = tmp_path / "deniable"
    outside.mkdir()
    prompt_count = 0

    async def _count_and_deny(directory: Path) -> bool:
        nonlocal prompt_count
        prompt_count += 1
        return False

    trust_mgr.set_prompt(_count_and_deny)

    async def _run_both():
        # First call: prompts and denies
        await trust_mgr.check_and_prompt(outside / "a.txt", workspace)
        # Second call: same trust target, should not re-prompt
        await trust_mgr.check_and_prompt(outside / "b.txt", workspace)

    asyncio.run(_run_both())
    assert prompt_count == 1


def test_state_file_content(state_dir: Path, tmp_path: Path):
    """The trusted_dirs.json file has the expected format."""
    mgr = TrustManager(state_dir)
    project = tmp_path / "project"
    project.mkdir()
    mgr.trust(project)

    state_file = state_dir / "trusted_dirs.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "directories" in data
    assert str(project.resolve()) in data["directories"]


def test_corrupted_state_file(state_dir: Path):
    """TrustManager handles corrupted state file gracefully."""
    (state_dir / "trusted_dirs.json").write_text("not valid json", encoding="utf-8")
    mgr = TrustManager(state_dir)
    assert mgr.trusted_directories == []
