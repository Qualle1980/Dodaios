"""An order already running must not start a second worker (phase 6).

The spawn choke point suppresses a duplicate only while a dispatch is in
flight AND within ``_COOLDOWN_SECONDS`` (30 s). That window was sized for the
VAD-fragment regression of 2026-05-27, where one spoken sentence arrived as
three turns. It does nothing for the other duplicate: a mission runs for
minutes, the person says the same thing again because nothing visibly
happened, and a SECOND worker starts on the same task — two worktrees, two
subscription turns, two critic loops, and two diffs for one piece of work.

The guard here asks the mission store instead of the clock: if a non-terminal
mission already names this work, the turn gets the same deterministic
"already running" ACK the cooldown returns. A different order still
dispatches while the first one runs, which is what the typed-order exception
in the cooldown exists for.

Deterministic — no worker, no model, no network.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.protocols import ExecutionContext
from jarvis.plugins.tool.spawn_worker import _COOLDOWN_SECONDS, SpawnWorkerTool

_RUNNING = "bau mir eine landingpage fuer das produkt"  # i18n-allow


class _Store:
    def __init__(self, missions: list[tuple[str, str, str]]) -> None:
        self._missions = missions
        self.queries = 0

    async def list_non_terminal_missions(self) -> list[tuple[str, str, str]]:
        self.queries += 1
        return list(self._missions)


class _Manager:
    def __init__(self, missions: list[tuple[str, str, str]] | None = None) -> None:
        self.dispatch_calls: list[dict[str, Any]] = []
        self.store = _Store(missions or [])

    async def dispatch(self, *, prompt: str, language: str, source_actor: str) -> str:
        self.dispatch_calls.append({"prompt": prompt, "language": language})
        return f"mission_{len(self.dispatch_calls):04d}"


def _ctx() -> ExecutionContext:
    return ExecutionContext(trace_id=uuid4(), user_utterance="", config={}, memory_read=None)


def _tool(manager: _Manager) -> SpawnWorkerTool:
    tool = SpawnWorkerTool(bus=EventBus(), manager=manager)
    # Well outside the cooldown window, and nothing in flight: the clock-based
    # guard cannot be what decides any of these cases.
    tool._last_spawn_at = time.monotonic() - (_COOLDOWN_SECONDS + 5.0)
    tool._active_dispatches = 0
    return tool


@pytest.mark.asyncio
async def test_the_same_order_does_not_start_a_second_worker() -> None:
    mgr = _Manager([("m1", _RUNNING, "RUNNING")])
    tool = _tool(mgr)

    result = await tool.execute({"utterance": _RUNNING, "action": ""}, _ctx())
    await asyncio.sleep(0.05)  # a dispatch, had one started, would land here

    assert mgr.dispatch_calls == [], "a running mission already covers this work"
    assert result.success is True
    assert any(
        isinstance(a, dict) and a.get("already_running_mission") == "m1"
        for a in (result.artifacts or ())
    ), f"the ACK must name the mission it deferred to, got {result.artifacts!r}"


@pytest.mark.asyncio
async def test_a_different_order_still_dispatches_while_one_runs() -> None:
    mgr = _Manager([("m1", _RUNNING, "RUNNING")])
    tool = _tool(mgr)

    await tool.execute(
        {"utterance": "pruefe die datenbank migration auf fehler", "action": ""},  # i18n-allow
        _ctx(),
    )
    await asyncio.sleep(0.05)  # the dispatch is fire-and-forget

    assert len(mgr.dispatch_calls) == 1


@pytest.mark.asyncio
async def test_a_finished_mission_does_not_block_a_repeat() -> None:
    """Only non-terminal missions are listed, so a redo of finished work runs."""
    mgr = _Manager([])
    tool = _tool(mgr)

    await tool.execute({"utterance": _RUNNING, "action": ""}, _ctx())
    await asyncio.sleep(0.05)  # the dispatch is fire-and-forget

    assert len(mgr.dispatch_calls) == 1


@pytest.mark.asyncio
async def test_a_manager_without_a_store_still_dispatches() -> None:
    """Before the mission stack is up there is no mission to collide with."""
    mgr = _Manager()
    del mgr.store
    tool = _tool(mgr)

    await tool.execute({"utterance": _RUNNING, "action": ""}, _ctx())
    await asyncio.sleep(0.05)  # the dispatch is fire-and-forget

    assert len(mgr.dispatch_calls) == 1
