"""Reuse of task worktrees: isolation, leases, crash recovery, the main checkout.

Real git repos under ``tmp_path`` instead of fakes, because every claim worth
making here is about what git actually does with a worktree — a mocked
``subprocess.run`` would prove only that the arguments were spelled as
expected. No model call, no mission, no network.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import threading
from pathlib import Path

import pytest

from jarvis.missions.isolation.workspace_pool import (
    LEASE_NAME,
    WorkspacePool,
    resolve_pool_root,
)
from jarvis.missions.isolation.worktree import (
    BASE_SHA_SIDECAR,
    WorktreeManager,
    read_worktree_base_sha,
)

_GIT_AVAILABLE = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not in PATH")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return (out.stdout or "").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A mini repo with two commits on ``main``."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "README.md").write_text("one\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "one")
    (root / "second.txt").write_text("two\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "two")
    return root


@pytest.fixture
def manager(repo: Path, tmp_path: Path) -> WorktreeManager:
    """A manager with one pool slot, outputs under ``tmp_path/outputs``."""
    return WorktreeManager(
        repo_root=repo, outputs_root=tmp_path / "outputs", pool_size=1,
    )


def _head(workspace: Path) -> str:
    return _git(workspace, "rev-parse", "HEAD")


# --- the slot's lifecycle ---------------------------------------------------


def test_a_finished_worktree_becomes_the_slot_and_the_next_task_reuses_it(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """Nothing is pre-built: the first task's tree is kept, the second reuses it."""
    pool_root = resolve_pool_root(tmp_path / "outputs")
    assert not pool_root.exists()  # no boot work, no speculative slot

    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)

    assert (pool_root / "ws-00" / "workspace" / "README.md").is_file()
    assert not first.exists()

    second = manager.create(mission_slug="m", task_id="02-b")
    assert second == pool_root / "ws-00" / "workspace"
    assert (second / "README.md").is_file()
    assert (second / "second.txt").is_file()


def test_the_reused_tree_is_a_pristine_checkout_of_the_repo_head(
    manager: WorktreeManager, repo: Path,
) -> None:
    """A worker's leftovers must not reach the next task, and HEAD must be the
    repo's commit — not whatever the previous task left in the slot."""
    first = manager.create(mission_slug="m", task_id="01-a")
    (first / "junk.txt").write_text("worker leftover\n", encoding="utf-8")
    (first / "README.md").write_text("worker edited this\n", encoding="utf-8")
    _git(first, "commit", "-am", "a commit the worker made")
    manager.remove(first, force=True)

    second = manager.create(mission_slug="m", task_id="02-b")
    assert not (second / "junk.txt").exists()
    assert (second / "README.md").read_text(encoding="utf-8") == "one\n"
    assert _head(second) == _git(repo, "rev-parse", "HEAD")
    assert _git(second, "status", "--porcelain") == ""


def test_the_base_sha_sidecar_names_the_commit_the_slot_was_reset_to(
    manager: WorktreeManager, repo: Path,
) -> None:
    """The Kontrollierer diffs a task against this SHA. A reused tree that
    reported the previous task's base would capture the wrong diff."""
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)

    (repo / "third.txt").write_text("three\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "three")
    moved_head = _git(repo, "rev-parse", "HEAD")

    second = manager.create(mission_slug="m", task_id="02-b")
    assert (second.parent / BASE_SHA_SIDECAR).is_file()
    assert read_worktree_base_sha(second) == moved_head
    assert _head(second) == moved_head
    assert (second / "third.txt").is_file()


def test_the_task_gets_its_own_branch_and_branches_do_not_pile_up(
    manager: WorktreeManager, repo: Path,
) -> None:
    first = manager.create(mission_slug="m", task_id="01-a")
    first_branch = _git(first, "symbolic-ref", "--short", "HEAD")
    manager.remove(first, force=True)
    second = manager.create(mission_slug="m", task_id="02-b")
    second_branch = _git(second, "symbolic-ref", "--short", "HEAD")

    assert first_branch.startswith("agent/")
    assert second_branch.startswith("agent/")
    assert second_branch != first_branch
    assert first_branch not in _git(repo, "branch", "--list").replace("*", "")


# --- isolation and exclusivity ---------------------------------------------


def test_two_tasks_never_share_a_slot(manager: WorktreeManager) -> None:
    """The second task runs while the first still holds the lease, so it must
    get a workspace of its own — a fresh checkout, not the leased slot."""
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)  # becomes the slot

    leased = manager.create(mission_slug="m", task_id="02-b")
    concurrent = manager.create(mission_slug="m", task_id="03-c")

    assert concurrent != leased
    assert (concurrent / "README.md").is_file()
    assert (leased / "README.md").is_file()


def test_a_lease_is_exclusive_across_threads(
    repo: Path, tmp_path: Path,
) -> None:
    """One slot, many threads: exactly one of them may be handed it."""
    manager = WorktreeManager(
        repo_root=repo, outputs_root=tmp_path / "outputs", pool_size=1,
    )
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)

    pool = WorkspacePool(
        repo_root=repo,
        pool_root=resolve_pool_root(tmp_path / "outputs"),
        size=1,
    )
    head = _git(repo, "rev-parse", "HEAD")
    results: list[Path | None] = []
    guard = threading.Lock()
    start = threading.Barrier(4)

    def worker(index: int) -> None:
        start.wait()
        got = pool.acquire(base_ref=head, branch=f"agent/t-{index}")
        with guard:
            results.append(got)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len([r for r in results if r is not None]) == 1


def test_a_lease_is_exclusive_across_processes(
    manager: WorktreeManager, tmp_path: Path, repo: Path,
) -> None:
    """Two app instances share one checkout. The lease is a file created
    O_EXCL, so a lease taken by a LIVE pid is not handed out again — an
    in-process set would not have caught this."""
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)
    slot = resolve_pool_root(tmp_path / "outputs") / "ws-00"
    # A lease owned by a process that is definitely alive: this one.
    (slot / LEASE_NAME).write_text(f"{os.getpid()}\n0\n", encoding="utf-8")

    second = manager.create(mission_slug="m", task_id="02-b")
    assert second != slot / "workspace"
    assert (slot / LEASE_NAME).is_file()  # untouched


# --- crash recovery ---------------------------------------------------------


def test_a_crashed_owners_slot_is_quarantined_with_its_files_not_reset(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """A dead lease owner means a task that may never have archived its
    results. Resetting that tree would delete the user's work, so the slot is
    set aside intact and the next task gets a fresh checkout."""
    pool_root = resolve_pool_root(tmp_path / "outputs")
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)
    slot = pool_root / "ws-00"
    (slot / "workspace" / "deliverable.html").write_text("work\n", encoding="utf-8")
    # pid 2**31-1 is not a live process on any supported platform.
    (slot / LEASE_NAME).write_text("2147483647\n0\n", encoding="utf-8")

    second = manager.create(mission_slug="m", task_id="02-b")

    assert second != slot / "workspace"
    assert not slot.exists()
    quarantined = [p for p in pool_root.iterdir() if p.name.startswith("quarantine-")]
    assert len(quarantined) == 1
    assert (quarantined[0] / "workspace" / "deliverable.html").read_text(
        encoding="utf-8"
    ) == "work\n"


def test_a_slot_that_is_not_a_worktree_of_this_repo_is_never_reset(
    repo: Path, tmp_path: Path,
) -> None:
    """The guard that keeps ``reset --hard`` away from a real checkout: a
    directory whose ``.git`` is a directory (a main checkout, or a lean
    workspace) is quarantined, never cleaned."""
    pool_root = tmp_path / "pool"
    slot = pool_root / "ws-00"
    workspace = slot / "workspace"
    workspace.mkdir(parents=True)
    _git(workspace, "init", "-b", "main")
    (workspace / "precious.txt").write_text("not ours\n", encoding="utf-8")

    pool = WorkspacePool(repo_root=repo, pool_root=pool_root, size=1)
    assert pool.acquire(base_ref="HEAD", branch="agent/x") is None
    quarantined = [p for p in pool_root.iterdir() if p.name.startswith("quarantine-")]
    assert len(quarantined) == 1
    assert (quarantined[0] / "workspace" / "precious.txt").is_file()


def test_the_pool_refuses_a_workspace_outside_its_own_root(
    repo: Path, tmp_path: Path,
) -> None:
    """``release`` and ``owns`` answer only for paths this pool leased, so a
    stray path can never route the caller's teardown into the pool."""
    pool = WorkspacePool(repo_root=repo, pool_root=tmp_path / "pool", size=1)
    assert pool.owns(repo) is False
    assert pool.release(repo) is False
    assert pool.adopt(repo) is False
    assert (repo / "README.md").read_text(encoding="utf-8") == "one\n"


# --- disabled, lean, and failure paths -------------------------------------


def test_pool_size_zero_is_exactly_the_old_behaviour(
    repo: Path, tmp_path: Path,
) -> None:
    manager = WorktreeManager(
        repo_root=repo, outputs_root=tmp_path / "outputs", pool_size=0,
    )
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)
    assert not first.exists()
    assert not resolve_pool_root(tmp_path / "outputs").exists()


def test_a_lean_workspace_is_never_adopted(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """A lean workspace is a standalone repo with no git relationship to this
    checkout; reusing one would hand the next task another mission's files."""
    lean = manager.create(mission_slug="m", task_id="01-a", needs_repo=False)
    manager.remove(lean, force=True)
    assert not lean.exists()
    assert not resolve_pool_root(tmp_path / "outputs").exists()


def test_a_broken_slot_makes_the_caller_fall_back_instead_of_failing(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """The pool is an optimisation and may never be the reason a task fails."""
    pool_root = resolve_pool_root(tmp_path / "outputs")
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)
    # Break the worktree link file: every git call in the slot now fails.
    # git leaves it read-only on Windows, so replace it rather than write it.
    link = pool_root / "ws-00" / "workspace" / ".git"
    link.chmod(stat.S_IWRITE | stat.S_IREAD)
    link.unlink()
    link.write_text("gitdir: /nowhere/at/all\n", encoding="utf-8")

    second = manager.create(mission_slug="m", task_id="02-b")
    assert (second / "README.md").is_file()
    assert second != pool_root / "ws-00" / "workspace"


def test_release_leaves_the_tree_for_the_next_acquire_to_reset(
    manager: WorktreeManager, tmp_path: Path,
) -> None:
    """Teardown stays instant — a cancelled mission waits on it — and a crash
    between release and the next acquire leaves the worker's files on disk."""
    slot = resolve_pool_root(tmp_path / "outputs") / "ws-00"
    first = manager.create(mission_slug="m", task_id="01-a")
    manager.remove(first, force=True)
    leased = manager.create(mission_slug="m", task_id="02-b")
    (leased / "unarchived.txt").write_text("still here\n", encoding="utf-8")

    manager.remove(leased, force=True)

    assert not (slot / LEASE_NAME).exists()
    assert (slot / "workspace" / "unarchived.txt").is_file()
