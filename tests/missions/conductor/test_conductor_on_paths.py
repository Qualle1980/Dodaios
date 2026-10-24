"""missions.conductor = "on": every path an order can take, on the real Kontrollierer.

Workers, critic and worktrees are the shared fakes; the conductor seat is
scripted. Covered: read orders stay read-only whatever the conductor writes,
the claude-code executor routes to the Claude worker, a spent subscription
parks the mission and the resume finishes it without redoing approved work,
a seat failure fails honestly, and no seat means the decomposer.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jarvis.missions.capacity import WorkerCapacityUnavailable, read_checkpoint
from jarvis.missions.kontrollierer.decomposer import MissionPlan, Step
from jarvis.missions.kontrollierer.worker_prompt import READ_ONLY_MARKER
from jarvis.missions.manager import MissionManager
from jarvis.missions.state_machine import MissionState
from tests.fakes.fake_mission_runtime import FakeMissionWorker, make_kontrollierer

_PLAN = MissionPlan(steps=[Step(slug="decomposed", prompt="decomposed step")], n_workers=1)


@pytest.fixture
async def manager(tmp_missions_db: Path):
    m = MissionManager(tmp_missions_db)
    await m.start()
    yield m
    await m.stop()


def _ledger(*, done: bool, nxt: str = "codex", instruction: str = "") -> str:
    return json.dumps({
        "is_request_satisfied": {"reason": "r", "answer": done},
        "is_in_loop": {"reason": "r", "answer": False},
        "is_progress_being_made": {"reason": "r", "answer": True},
        "next_executor": {"reason": "r", "answer": nxt},
        "instruction": {"reason": "r", "answer": instruction},
    })


class _Seat:
    def __init__(self, *ledgers: str, fail_on_ledger: int | None = None) -> None:
        self.ledgers = list(ledgers)
        self.fail_on_ledger = fail_on_ledger
        self.seen = 0

    async def __call__(self, messages: Sequence[tuple[str, str]]) -> str:
        prompt = messages[-1][1]
        if "Recall the request" in prompt:
            self.seen += 1
            if self.fail_on_ledger == self.seen:
                raise RuntimeError("seat went away")
            return self.ledgers.pop(0)
        if "We worked on this request" in prompt:
            return "Final answer."
        return "- codex: do it"


class _Factory:
    """Records each step; ``park_from`` makes the n-th build hit a spent window."""

    def __init__(self, park_from: int | None = None) -> None:
        self.steps: list[Step] = []
        self.workers: list[FakeMissionWorker] = []
        self.park_from = park_from

    def __call__(self, step: Step) -> FakeMissionWorker:
        if self.park_from is not None and len(self.steps) + 1 >= self.park_from:
            raise WorkerCapacityUnavailable("provider_quota", "codex")
        self.steps.append(step)
        worker = FakeMissionWorker(family="codex")
        self.workers.append(worker)
        return worker


def _kontrollierer(manager: MissionManager, tmp_path: Path, factory: Any, seat: Any) -> Any:
    k = make_kontrollierer(manager, tmp_path, _PLAN, factory)

    async def _seat_of() -> Any:
        if isinstance(seat, Exception):
            raise seat
        return seat

    k.set_conductor(lambda: "on", _seat_of)
    return k


async def test_a_read_order_stays_read_only_whatever_the_conductor_writes(
    manager: MissionManager, tmp_path: Path
) -> None:
    factory = _Factory()
    seat = _Seat(_ledger(done=False, instruction="Run `wc -l a.py b.py` and report"),
                 _ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="How many lines do a.py and b.py each have?")

    assert await k.run_mission(mid) == MissionState.APPROVED
    prompt = factory.workers[0].prompts[0]
    assert prompt.lstrip().startswith(READ_ONLY_MARKER)  # -> codex --sandbox read-only


async def test_a_change_order_keeps_write_access(manager: MissionManager, tmp_path: Path) -> None:
    factory = _Factory()
    seat = _Seat(_ledger(done=False, instruction="Create docs/a.md with one line"),
                 _ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="Create the file docs/a.md containing 'probe'.")

    assert await k.run_mission(mid) == MissionState.APPROVED
    assert not factory.workers[0].prompts[0].lstrip().startswith(READ_ONLY_MARKER)


async def test_the_claude_code_executor_routes_to_the_claude_worker(
    manager: MissionManager, tmp_path: Path
) -> None:
    factory = _Factory()
    seat = _Seat(_ledger(done=False, nxt="claude-code", instruction="Refactor the router"),
                 _ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="Refactor the router across the codebase")

    assert await k.run_mission(mid) == MissionState.APPROVED
    assert [(s.worker_cli, s.preferred_cli) for s in factory.steps] == [("claude", "claude")]


async def test_a_spent_window_parks_and_the_resume_finishes_without_redoing_work(
    manager: MissionManager, tmp_path: Path
) -> None:
    factory = _Factory(park_from=2)
    seat = _Seat(_ledger(done=False, instruction="Step one"),
                 _ledger(done=False, instruction="Step two"))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="Create two files docs/a.md and docs/b.md")

    assert await k.run_mission(mid) == MissionState.WAITING_CAPACITY
    assert seat.seen == 2  # parked at once: no further progress call on the seat
    checkpoint = read_checkpoint(tmp_path / "missions" / f"mission_{mid[:13]}")
    assert checkpoint is not None
    approved_first = factory.steps[0].task_id

    factory.park_from = None  # capacity is back
    assert await k.resume_mission(mid) == MissionState.APPROVED
    built = [s.task_id for s in factory.steps]
    assert built.count(approved_first) == 1  # the approved step never ran again
    assert [s.prompt for s in factory.steps][-1].endswith("Step two")


async def test_a_seat_failure_mid_order_fails_honestly(
    manager: MissionManager, tmp_path: Path
) -> None:
    factory = _Factory()
    seat = _Seat(_ledger(done=False, instruction="Step one"), fail_on_ledger=2)
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="Create docs/a.md")

    assert await k.run_mission(mid) == MissionState.FAILED
    failed = [e.payload for e in await manager.store.events_for_mission(mid)
              if e.payload.event_type == "MissionFailed"]
    assert failed and failed[0].reason.startswith("conductor_failed")


@pytest.mark.parametrize("seat", [None, RuntimeError("probe crashed")])
async def test_no_usable_seat_takes_the_decomposer(
    manager: MissionManager, tmp_path: Path, seat: Any
) -> None:
    factory = _Factory()
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="Create docs/a.md")

    assert await k.run_mission(mid) == MissionState.APPROVED
    assert [s.prompt for s in factory.steps] == ["decomposed step"]
