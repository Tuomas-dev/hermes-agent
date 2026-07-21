"""Real-Git behavior tests for fail-safe managed updates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.main import _classify_git_branch_relation


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def repositories(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    _git(tmp_path, "init", "-q", "-b", "main", str(seed))
    for repo in (seed,):
        _git(repo, "config", "user.email", "tests@example.com")
        _git(repo, "config", "user.name", "Hermes tests")
    (seed / "history.txt").write_text("base\n")
    _git(seed, "add", "history.txt")
    _git(seed, "commit", "-qm", "base")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-q", "-u", "origin", "main")
    _git(tmp_path, "clone", "-q", "-b", "main", str(remote), str(local))
    _git(local, "config", "user.email", "tests@example.com")
    _git(local, "config", "user.name", "Hermes tests")
    return remote, seed, local


def _relation(local: Path) -> tuple[str, int, int]:
    _git(local, "fetch", "-q", "origin", "main")
    return _classify_git_branch_relation(["git"], local, "main", "origin/main")


def test_equal_history(repositories) -> None:
    _remote, _seed, local = repositories
    assert _relation(local) == ("equal", 0, 0)


def test_strictly_behind_history(repositories) -> None:
    _remote, seed, local = repositories
    (seed / "history.txt").write_text("base\nremote\n")
    _git(seed, "commit", "-qam", "remote")
    _git(seed, "push", "-q")
    assert _relation(local) == ("behind", 0, 1)


def test_strictly_ahead_history_preserves_local_commit(repositories) -> None:
    _remote, _seed, local = repositories
    (local / "local.txt").write_text("production fix\n")
    _git(local, "add", "local.txt")
    _git(local, "commit", "-qm", "local fix")
    local_sha = _git(local, "rev-parse", "HEAD")

    assert _relation(local) == ("ahead", 1, 0)
    assert _git(local, "rev-parse", "HEAD") == local_sha


def test_diverged_history_preserves_local_commit(repositories) -> None:
    _remote, seed, local = repositories
    (local / "local.txt").write_text("production fix\n")
    _git(local, "add", "local.txt")
    _git(local, "commit", "-qm", "local fix")
    local_sha = _git(local, "rev-parse", "HEAD")
    (seed / "remote.txt").write_text("upstream fix\n")
    _git(seed, "add", "remote.txt")
    _git(seed, "commit", "-qm", "remote fix")
    _git(seed, "push", "-q")

    assert _relation(local) == ("diverged", 1, 1)
    assert _git(local, "rev-parse", "HEAD") == local_sha
