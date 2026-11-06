"""A rejection the gate speaks is also written down (Phase C1).

The gate's own contract is that it grants nothing and retries nothing. These
tests pin the one thing that was missing next to it: the failure reaches the
incident record, and the conductor's seat is asked at most once per job.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from jarvis.agent_chat.delegation_lessons import document_rejection
from jarvis.missions.conductor.incidents import IncidentRecorder, set_active_recorder
from jarvis.missions.conductor.lessons import LessonStore

ORDER = "Add subtract() to the calculator and run the tests."


@dataclass(frozen=True)
class FakeJob:
    task_id: str
    status: str = "completed"
    workspace_id: str = "ws-1"


@dataclass(frozen=True)
class FakeGate:
    """Only what ``document_rejection`` reads: the jobs, the work, the gaps."""

    jobs: tuple[FakeJob, ...]
    missing: tuple[str, ...] = ("no test run inside the job's checkout",)

    @property
    def work(self) -> tuple[str, ...]:
        return tuple(job.task_id for job in self.jobs)

    def gaps(self, _ledger: object) -> tuple[str, ...]:
        return self.missing


def _lines(root: Path) -> list[dict]:
    path = root / "incidents.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


async def test_a_rejection_is_recorded_with_the_missing_evidence(tmp_path: Path) -> None:
    recorder = IncidentRecorder(LessonStore(tmp_path))
    set_active_recorder(recorder)
    try:
        assert await document_rejection(FakeGate((FakeJob("coding-1"),)), None, ORDER) is True
    finally:
        set_active_recorder(None)

    rows = _lines(tmp_path)
    assert len(rows) == 1, rows
    assert "no test run" in rows[0]["cause"]
    assert "coding-1" in rows[0]["cause"]
    assert rows[0]["executor"] == "ws-1"
    assert rows[0]["request"].startswith("Add subtract()")


async def test_a_crashed_handback_is_documented_too(tmp_path: Path) -> None:
    """A crashed or blocked agent is further from acceptance, not closer to it
    (``gate_for``), so it arms the gate as well -- and it is worth writing down.
    What must NOT happen here is a retry, and nothing in this path dispatches
    one."""
    recorder = IncidentRecorder(LessonStore(tmp_path))
    set_active_recorder(recorder)
    try:
        gate = FakeGate((FakeJob("coding-2", status="error"),), missing=("no change read",))
        assert await document_rejection(gate, None, ORDER) is True
    finally:
        set_active_recorder(None)

    rows = _lines(tmp_path)
    assert len(rows) == 1
    assert "reported error" in rows[0]["cause"], rows[0]["cause"]


async def test_the_seat_is_asked_once_per_job_not_once_per_rejection(tmp_path: Path) -> None:
    asked: list[str] = []

    async def _seat(prompt: str) -> str:
        asked.append(prompt)
        return "ADD 1: Run the tests inside the job's own checkout before claiming a pass."

    recorder = IncidentRecorder(LessonStore(tmp_path), learn=_seat)
    set_active_recorder(recorder)
    try:
        gate = FakeGate((FakeJob("coding-3"),))
        await document_rejection(gate, None, ORDER)
        await document_rejection(gate, None, ORDER)
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        set_active_recorder(None)

    assert len(asked) == 1, asked
    assert len(_lines(tmp_path)) == 2  # both rejections documented all the same


async def test_without_a_recorder_the_turn_is_not_disturbed(tmp_path: Path) -> None:
    """Before the mission runtime has booted there is nothing to document into.
    That is a no-op, never an exception into the chat turn."""
    set_active_recorder(None)
    assert await document_rejection(FakeGate((FakeJob("coding-4"),)), None, ORDER) is False
    assert _lines(tmp_path) == []


async def test_a_gate_that_raises_does_not_break_the_answer(tmp_path: Path) -> None:
    class Exploding(FakeGate):
        def gaps(self, _ledger: object) -> tuple[str, ...]:
            raise RuntimeError("ledger gone")

    recorder = IncidentRecorder(LessonStore(tmp_path))
    set_active_recorder(recorder)
    try:
        assert await document_rejection(Exploding((FakeJob("coding-5"),)), None, ORDER) is False
    finally:
        set_active_recorder(None)
    assert _lines(tmp_path) == []
