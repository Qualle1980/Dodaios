"""The worktree lock under cancellation. No git, no mission, no model call.

A thread already blocked on the process-wide worktree lock cannot be
interrupted, so a cancelled step used to keep going: it acquired the lock,
built a whole worktree nobody would ever read, and held the lock while the next
task queued behind it. That is invisible setup time charged to the wrong task.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path

import pytest

from jarvis.missions.kontrollierer.orchestrator import Kontrollierer
from jarvis.missions.timing import timed_offload


class _Recorder:
    """Stands in for ``WorktreeManager``; records whether it was asked to build."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs) -> Path:
        self.calls.append(kwargs)
        return Path("workspace")


@pytest.fixture
def orchestrator() -> Kontrollierer:
    orch = Kontrollierer.__new__(Kontrollierer)
    orch._worktrees = _Recorder()
    return orch


@pytest.fixture(autouse=True)
def _lock_is_free_again():
    """A leaked lock would hang every later case, so prove it was released."""
    yield
    acquired = Kontrollierer._worktree_lock.acquire(timeout=5)
    assert acquired, "the worktree lock was not released"
    Kontrollierer._worktree_lock.release()


def test_an_abandoned_setup_skips_the_checkout(orchestrator: Kontrollierer) -> None:
    abandoned = threading.Event()
    abandoned.set()

    with pytest.raises(asyncio.CancelledError):
        orchestrator._create_worktree(task_id="t-1", abandoned=abandoned)

    assert orchestrator._worktrees.calls == []


def test_a_live_setup_still_builds_its_workspace(orchestrator: Kontrollierer) -> None:
    result = orchestrator._create_worktree(
        task_id="t-2", abandoned=threading.Event(), needs_repo=True
    )

    assert result == Path("workspace")
    assert orchestrator._worktrees.calls == [{"task_id": "t-2", "needs_repo": True}]


def test_a_setup_without_the_flag_behaves_as_before(
    orchestrator: Kontrollierer,
) -> None:
    """The argument is optional, so an older caller is unaffected."""
    assert orchestrator._create_worktree(task_id="t-3") == Path("workspace")
    assert orchestrator._worktrees.calls == [{"task_id": "t-3"}]


@pytest.mark.asyncio
async def test_cancelling_the_await_stops_the_queued_thread_building_an_orphan(
    orchestrator: Kontrollierer,
) -> None:
    """The real race: cancelled while the thread waits for another task's lock.

    Measured before the fix: the checkout ran anyway, after the caller was
    already gone.
    """
    abandoned = threading.Event()
    assert Kontrollierer._worktree_lock.acquire(timeout=5)
    released = False

    async def setup() -> Path:
        try:
            return await timed_offload(
                "setup.worktree_offloaded",
                "t-4",
                orchestrator._create_worktree,
                task_id="t-4",
                abandoned=abandoned,
            )
        except asyncio.CancelledError:
            abandoned.set()
            raise

    task = asyncio.create_task(setup())
    try:
        # Let the worker thread reach the lock, then abandon the await.
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert orchestrator._worktrees.calls == []
    finally:
        Kontrollierer._worktree_lock.release()
        released = True

    # Give the abandoned thread every chance to do the work anyway.
    for _ in range(40):
        if orchestrator._worktrees.calls:
            break
        await asyncio.sleep(0.05)

    assert released
    assert orchestrator._worktrees.calls == [], "a cancelled step built an orphan"
