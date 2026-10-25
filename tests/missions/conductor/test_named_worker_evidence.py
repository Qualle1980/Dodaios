"""An order that names its worker is not done until that worker has run.

Live 2026-10-25, the German order "Lass Codex nachsehen, wie viele Zeilen
jarvis/missions/conductor/loop.py hat." reached no worker at all: the
conductor answered from the seat, where it cannot open a file. Two things
had to change — the named worker becomes the only executor, and a seat
answer no longer counts as the work being done.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jarvis.missions.kontrollierer.decomposer import MissionPlan, Step
from jarvis.missions.kontrollierer.worker_prompt import READ_ONLY_MARKER
from jarvis.missions.manager import MissionManager
from jarvis.missions.state_machine import MissionState
from tests.fakes.fake_mission_runtime import FakeMissionWorker, make_kontrollierer

LIVE = "Lass Codex nachsehen, wie viele Zeilen jarvis/missions/conductor/loop.py hat."

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
    """Records the team roster the conductor was handed."""

    def __init__(self, *ledgers: str) -> None:
        self.ledgers = list(ledgers)
        self.rosters: list[str] = []

    async def __call__(self, messages: Sequence[tuple[str, str]]) -> str:
        prompt = messages[-1][1]
        if "codex:" in prompt or "conductor:" in prompt:
            self.rosters.append(prompt)
        if "Recall the request" in prompt:
            return self.ledgers.pop(0)
        if "We worked on this request" in prompt:
            return "252"
        return "- codex: do it"


class _Factory:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.workers: list[FakeMissionWorker] = []

    def __call__(self, step: Step) -> FakeMissionWorker:
        self.steps.append(step)
        worker = FakeMissionWorker(family="codex")
        self.workers.append(worker)
        return worker


def _kontrollierer(manager: MissionManager, tmp_path: Path, factory: Any, seat: Any) -> Any:
    k = make_kontrollierer(manager, tmp_path, _PLAN, factory)

    async def _seat_of() -> Any:
        return seat

    k.set_conductor(lambda: "on", _seat_of)
    return k


async def test_the_live_order_reaches_codex_read_only(
    manager: MissionManager, tmp_path: Path
) -> None:
    """The exact order that failed live, start to finish."""
    factory = _Factory()
    seat = _Seat(_ledger(done=False, instruction="Count the lines of loop.py"),
                 _ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt=LIVE)

    assert await k.run_mission(mid) == MissionState.APPROVED
    # It went to Codex, not to the conductor's own seat answer.
    assert [(s.worker_cli, s.preferred_cli) for s in factory.steps] == [("codex", "codex")]
    # And it was classified as a read, so the worker runs --sandbox read-only.
    assert factory.workers[0].prompts[0].lstrip().startswith(READ_ONLY_MARKER)


async def test_the_conductor_is_not_offered_as_a_substitute(
    manager: MissionManager, tmp_path: Path
) -> None:
    factory = _Factory()
    seat = _Seat(_ledger(done=False, instruction="Count the lines"), _ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    await k.run_mission(await manager.dispatch(prompt=LIVE))

    assert seat.rosters, "the conductor was never handed a team"
    for roster in seat.rosters:
        assert "conductor: you yourself" not in roster


async def test_a_named_order_answered_only_from_the_seat_fails(
    manager: MissionManager, tmp_path: Path
) -> None:
    """The live failure mode: no worker ran, so this is not an answer."""
    factory = _Factory()
    seat = _Seat(_ledger(done=True))  # satisfied on the first look, nobody ran
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt=LIVE)

    assert await k.run_mission(mid) == MissionState.FAILED
    assert factory.steps == []


async def test_an_unnamed_order_may_still_be_answered_from_the_seat(
    manager: MissionManager, tmp_path: Path
) -> None:
    """The evidence rule binds the named case only; ordinary routing is untouched."""
    factory = _Factory()
    seat = _Seat(_ledger(done=True))
    k = _kontrollierer(manager, tmp_path, factory, seat)
    mid = await manager.dispatch(prompt="What is a worktree?")

    assert await k.run_mission(mid) == MissionState.APPROVED
    assert factory.steps == []
