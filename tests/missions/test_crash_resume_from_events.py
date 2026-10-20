"""An interrupted mission keeps the steps it already finished.

Until 2026-10-20 a checkpoint existed only when a mission PARKED for capacity.
A graceful shutdown therefore reparked and resumed, while a hard crash left
nothing: recovery swept the mission ``crash_recovery`` and a re-run
re-dispatched the prompt as a NEW mission from zero. Every step that had
already passed review was built again, on the same subscription, and paid for
twice.

Nothing had to be written more often to fix that. The event log already holds
the plan (``MissionPlanReady``), which step each worker took
(``WorkerSpawned.step``) and which steps passed review (an approving
``CriticVerdictReady``) — so the checkpoint is replayed out of the log and the
existing, proven resume path continues from it.

Deterministic — no worker, no provider, no model.
"""
from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from jarvis.missions.events import (
    CriticVerdictReady,
    EventEnvelope,
    MissionPlanReady,
    MissionStateChanged,
    WorkerSpawned,
    now_ms,
)
from jarvis.missions.manager import MissionManager
from jarvis.missions.recovery import startup_recover
from jarvis.missions.replay import (
    CRASH_RESUME_REASON,
    MAX_CRASH_RESUMES,
    replay_checkpoint,
)
from jarvis.missions.state_machine import MissionState

_HOUR_MS = 3_600_000


@pytest_asyncio.fixture
async def store(tmp_missions_db: Path):
    manager = MissionManager(tmp_missions_db)
    await manager.start(recover=False)
    try:
        yield manager
    finally:
        await manager.stop()


def _step(task_id: str, slug: str) -> dict:
    return {"task_id": task_id, "slug": slug, "prompt": f"do {slug}"}


async def _interrupted_mission(
    manager: MissionManager,
    *,
    approved: tuple[str, ...],
    steps: tuple[str, ...] = ("t1", "t2"),
    age_ms: int = _HOUR_MS,
) -> str:
    """A RUNNING mission whose log shows ``approved`` steps and then stops."""
    mission_id = await manager.dispatch(prompt="build the thing")
    await manager.transition_state(mission_id, MissionState.RUNNING, reason="spawn")
    old = now_ms() - age_ms

    async def _append(payload) -> None:
        await manager.store.append_and_publish(
            EventEnvelope(
                mission_id=mission_id,
                source_actor="system",
                ts_ms=old,
                payload=payload,
            )
        )

    await _append(
        MissionPlanReady(
            plan=[_step(t, f"step-{t}") for t in steps],
            n_workers=len(steps),
            expected_output="a diff",
        )
    )
    for task_id in approved:
        worker_id = f"w-{task_id}"
        await _append(
            WorkerSpawned(
                worker_id=worker_id,
                step=_step(task_id, f"step-{task_id}"),
                pid=4242,
                cli="codex",
                model="",
                worktree="worktrees/step",
            )
        )
        await _append(
            CriticVerdictReady(
                worker_id=worker_id,
                verdict="approve",
                summary="looks right",
                confidence=0.9,
                iteration=0,
            )
        )
    return mission_id


# --------------------------------------------------------------------------
# The replay itself.
# --------------------------------------------------------------------------

async def test_the_log_alone_says_which_steps_are_done(store: MissionManager) -> None:
    mission_id = await _interrupted_mission(store, approved=("t1",))
    events = await store.store.events_for_mission(mission_id)

    checkpoint = replay_checkpoint(mission_id, "build the thing", events)

    assert checkpoint is not None
    assert checkpoint["steps"] == [
        {"task_id": "t1", "done": True},
        {"task_id": "t2", "done": False},
    ]
    assert checkpoint["provider"] is None, "a crash pins no provider"


async def test_nothing_approved_is_not_worth_resuming(store: MissionManager) -> None:
    mission_id = await _interrupted_mission(store, approved=())
    events = await store.store.events_for_mission(mission_id)

    assert replay_checkpoint(mission_id, "build the thing", events) is None


async def test_every_step_approved_is_a_finalization_not_a_resume(
    store: MissionManager,
) -> None:
    mission_id = await _interrupted_mission(store, approved=("t1", "t2"))
    events = await store.store.events_for_mission(mission_id)

    assert replay_checkpoint(mission_id, "build the thing", events) is None


# --------------------------------------------------------------------------
# What recovery does with it.
# --------------------------------------------------------------------------

async def test_an_interrupted_mission_is_parked_not_failed(
    store: MissionManager, tmp_path: Path
) -> None:
    mission_id = await _interrupted_mission(store, approved=("t1",))

    swept = await startup_recover(
        store.store, stale_after_ms=1, isolation_root=tmp_path
    )

    assert swept == [], "an interrupted mission with finished work is not a failure"
    view = await store.mission(mission_id)
    assert view is not None and view.state == MissionState.WAITING_CAPACITY
    checkpoint = tmp_path / f"mission_{mission_id[:13]}" / "checkpoint.json"
    assert checkpoint.exists(), "the resume path reads this file"


async def test_without_an_isolation_root_the_sweep_is_unchanged(
    store: MissionManager,
) -> None:
    """A caller that knows no output root loses nothing it had before."""
    mission_id = await _interrupted_mission(store, approved=("t1",))

    swept = await startup_recover(store.store, stale_after_ms=1)

    assert swept == [mission_id]
    view = await store.mission(mission_id)
    assert view is not None and view.state == MissionState.FAILED


async def test_a_mission_with_no_finished_work_still_fails(
    store: MissionManager, tmp_path: Path
) -> None:
    mission_id = await _interrupted_mission(store, approved=())

    swept = await startup_recover(
        store.store, stale_after_ms=1, isolation_root=tmp_path
    )

    assert swept == [mission_id]


async def test_a_mission_that_keeps_crashing_is_failed_honestly(
    store: MissionManager, tmp_path: Path
) -> None:
    """No automatic retry loop: a mission that takes the app down with it must
    not come back on every boot."""
    mission_id = await _interrupted_mission(store, approved=("t1",))
    old = now_ms() - _HOUR_MS
    for _ in range(MAX_CRASH_RESUMES):
        await store.store.append_and_publish(
            EventEnvelope(
                mission_id=mission_id,
                source_actor="system",
                ts_ms=old,
                payload=MissionStateChanged(
                    from_state="RUNNING",
                    to_state="WAITING_CAPACITY",
                    reason=CRASH_RESUME_REASON,
                ),
            )
        )

    swept = await startup_recover(
        store.store, stale_after_ms=1, isolation_root=tmp_path
    )

    assert swept == [mission_id]


async def test_the_replayed_plan_is_what_the_resume_path_accepts(
    store: MissionManager,
) -> None:
    """The contract between the replay and resume: same parser, same shape.

    ``Step`` forbids extra keys, so a plan rebuilt from the event payloads has
    to validate through the very helper ``resume_mission`` uses — otherwise the
    park would strand the mission with an unusable checkpoint.
    """
    from jarvis.missions.kontrollierer.orchestrator import (
        _done_task_ids,
        _plan_from_checkpoint,
    )

    mission_id = await _interrupted_mission(store, approved=("t1",))
    events = await store.store.events_for_mission(mission_id)
    checkpoint = replay_checkpoint(mission_id, "build the thing", events)
    assert checkpoint is not None

    plan = _plan_from_checkpoint(checkpoint)

    assert plan is not None
    assert [step.task_id for step in plan.steps] == ["t1", "t2"]
    assert _done_task_ids(checkpoint) == {"t1"}
