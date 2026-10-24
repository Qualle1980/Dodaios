"""Shadow mode records what the conductor would do, and never touches the order."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from jarvis.missions.conductor.lessons import LessonStore
from jarvis.missions.conductor.loop import Message
from jarvis.missions.conductor.shadow import ShadowLog, bind_conductor_shadow
from jarvis.missions.events import EventEnvelope, MissionDispatched, WorkerSpawned

_LEDGER = json.dumps({
    "is_request_satisfied": {"reason": "r", "answer": False},
    "is_in_loop": {"reason": "r", "answer": False},
    "is_progress_being_made": {"reason": "r", "answer": True},
    "next_executor": {"reason": "a file read", "answer": "codex"},
    "instruction": {"reason": "r", "answer": "rg -n TODO jarvis/missions/budget.py"},
})


class _Bus:
    def __init__(self) -> None:
        self.handlers: list = []

    def subscribe_all(self, handler) -> None:  # type: ignore[no-untyped-def]
        self.handlers.append(handler)

    async def emit(self, mission_id: str, payload) -> None:  # type: ignore[no-untyped-def]
        env = EventEnvelope(mission_id=mission_id, source_actor="system", ts_ms=1,
                            payload=payload)
        for handler in self.handlers:
            await handler(env)


def _bind(tmp_path: Path, mode: str, seat_calls: list[str], seat: bool = True) -> _Bus:
    async def _ask(messages: Sequence[Message]) -> str:
        prompt = messages[-1][1]
        seat_calls.append(prompt[:20])
        return _LEDGER if "Recall the request" in prompt else "- codex: search the file"

    async def _seat():  # type: ignore[no-untyped-def]
        return _ask if seat else None

    bus = _Bus()
    bind_conductor_shadow(bus, ShadowLog(tmp_path), LessonStore(tmp_path), lambda: mode, _seat)
    return bus


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def _records(tmp_path: Path) -> list[dict]:
    path = tmp_path / "shadow.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


async def test_shadow_previews_an_order_and_pairs_it_with_the_real_worker(tmp_path: Path) -> None:
    calls: list[str] = []
    bus = _bind(tmp_path, "shadow", calls)
    await bus.emit("m1", MissionDispatched(prompt="List the TODOs in jarvis/missions/budget.py"))
    await _settle()
    await bus.emit("m1", WorkerSpawned(worker_id="w", pid=1, cli="codex", model="gpt",
                                       worktree="x"))
    await bus.emit("m1", WorkerSpawned(worker_id="w2", pid=2, cli="claude", model="m",
                                       worktree="x"))

    preview, actual = _records(tmp_path)
    assert preview["kind"] == "preview" and preview["lane"] == "lookup"
    assert preview["conductor_executor"] == "codex"
    assert preview["conductor_instruction"].startswith("rg -n TODO")
    assert actual == {**actual, "kind": "actual", "executor": "codex"}  # first worker only
    assert len(calls) == 3  # facts, plan, one progress record


async def test_off_sub_missions_and_no_seat_cost_nothing(tmp_path: Path) -> None:
    calls: list[str] = []
    off = _bind(tmp_path / "off", "off", calls)
    await off.emit("m1", MissionDispatched(prompt="Fix the typo in README.md"))
    shadow = _bind(tmp_path / "sub", "shadow", calls)
    await shadow.emit("m2", MissionDispatched(prompt="Fix it", parent_mission_id="m1"))
    no_seat = _bind(tmp_path / "seat", "shadow", calls, seat=False)
    await no_seat.emit("m3", MissionDispatched(prompt="Fix the typo in README.md"))
    await _settle()

    assert calls == []
    assert _records(tmp_path / "off") == _records(tmp_path / "sub") == []
    assert _records(tmp_path / "seat") == []
