"""Shadow mode: the conductor plans every order, nothing it says steers it.

``missions.conductor = "shadow"`` (ADR-0043 rollout). For each top-level
:class:`MissionDispatched` the conductor runs :meth:`.loop.Conductor.preview`
(facts, plan, first progress record: three seat calls) in the background and
writes one line to ``shadow.jsonl``: the deterministic lane, the executor and
instruction the conductor would hand out first, and its plan. The first
:class:`WorkerSpawned` of the same mission adds the executor that really ran,
so the two can be compared before the conductor is allowed to steer.

The seat is a connected subscription or nothing (background policy): without
one the order is not previewed, and the mission itself is never touched.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

from ..events import EventEnvelope, MissionDispatched, WorkerSpawned
from ..stream_evidence import clean_request_body
from .lanes import first_lane
from .lessons import LessonStore
from .loop import Ask, Conductor

logger = logging.getLogger(__name__)

#: The executors the conductor may name, and what each is for (ADR-0043 lanes).
TEAM: Final[dict[str, str]] = {
    "conductor": "you yourself: answer knowledge questions, summaries and plans "
                 "without any worker (cheapest, fastest)",
    "codex": "reads, searches and edits files in the repository; small fixes "
             "and features (cheap)",
    "claude-code": "large refactors and review-heavy changes across many files",
}

#: A CLI name on WorkerSpawned -> the conductor's executor name.
_CLI_EXECUTOR: Final[dict[str, str]] = {"codex": "codex", "claude": "claude-code"}

ModeLookup = Callable[[], str]
SeatLookup = Callable[[], Awaitable[Ask | None]]


class ShadowLog:
    """``shadow.jsonl``: one line per preview and per first real worker."""

    def __init__(self, root: Path) -> None:
        self._path = root / "shadow.jsonl"
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps({**record, "ts_ms": int(time.time() * 1000)}, ensure_ascii=False)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            logger.warning("conductor shadow: could not write a record", exc_info=True)


def bind_conductor_shadow(
    bus: Any,
    log: ShadowLog,
    lessons: LessonStore,
    mode_of: ModeLookup,
    seat_of: SeatLookup,
) -> None:
    """Preview every top-level order and record the first real worker."""
    previewed: set[str] = set()
    actual_seen: set[str] = set()
    pending: set[asyncio.Task[None]] = set()

    async def _preview(mission_id: str, request: str) -> None:
        lane = first_lane(request)
        started = time.monotonic()
        try:
            ask = await seat_of()
            if ask is None:
                logger.info("conductor shadow: no subscription seat, %s not previewed", mission_id)
                return
            conductor = Conductor(ask, TEAM, lessons=lessons.render(lane))
            preview = await conductor.preview(request)
        except Exception:  # noqa: BLE001 - shadow mode must never cost the order
            logger.warning("conductor shadow: preview failed for %s", mission_id, exc_info=True)
            return
        first = preview.first
        await asyncio.to_thread(log.write, {
            "kind": "preview", "mission_id": mission_id, "request": request[:500],
            "lane": lane,
            "conductor_executor": first.next_executor.answer if first else None,
            "conductor_instruction": first.instruction.answer[:500] if first else None,
            "satisfied_without_work": bool(first and first.is_request_satisfied.answer),
            "plan": preview.plan[:1500],
            "seconds": round(time.monotonic() - started, 1),
        })
        logger.info("conductor shadow: %s lane=%s conductor would start with %s",
                    mission_id, lane, first.next_executor.answer if first else "nothing")

    async def _handler(env: EventEnvelope) -> None:
        payload = env.payload
        if isinstance(payload, WorkerSpawned):
            if env.mission_id in previewed and env.mission_id not in actual_seen:
                actual_seen.add(env.mission_id)
                await asyncio.to_thread(log.write, {
                    "kind": "actual", "mission_id": env.mission_id,
                    "executor": _CLI_EXECUTOR.get(payload.cli, payload.cli),
                    "model": payload.model,
                })
            return
        if not isinstance(payload, MissionDispatched) or payload.parent_mission_id:
            return
        if mode_of() != "shadow" or env.mission_id in previewed:
            return
        request = clean_request_body(payload.prompt or "")
        if not request.strip():
            return
        previewed.add(env.mission_id)
        task = asyncio.get_running_loop().create_task(_preview(env.mission_id, request))
        pending.add(task)
        task.add_done_callback(pending.discard)

    bus.subscribe_all(_handler)


__all__ = ["TEAM", "ShadowLog", "bind_conductor_shadow"]
