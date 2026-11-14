"""Identical live tool calls in one turn must stop after the first execution.

2026-11-14 night sessions: the model called ``workspace-orchestrate`` (filter
on) or ``run_app_action`` (filter off) dozens of times at 2–3 calls/s with
``turn complete (function call, boundary withheld) — audio=0.0s``. Most
generations never spoke. Flight recorder showed many identical
``{"action": "status"}`` successes and opaque call-id collisions.

PR #142 fixed the empty-schema memory-tool cause of the 2026-11-09 177-call
loop but deliberately deferred a per-turn identical-call bound. This file is
that bound: progress (new args) still runs; repetition (same tool+args in the
same revision) returns the prior receipt and tells the model to speak.

These tests drive real ``LiveTools`` / ``NativeLiveVoiceSession`` over fakes
from ``tests/fakes/`` — they execute calls, they do not only assert the catalog.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jarvis.live import native
from jarvis.live.native import NativeLiveVoiceSession
from jarvis.live.state import LiveLedger
from jarvis.live.tools import LiveTools, _REPEAT_SPEAK_NOW
from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import (
    FakeToolGateway,
    gemini_like_provider,
    local_voice_like_provider,
)

PROVIDERS = {
    "gemini-like": gemini_like_provider,
    "local-voice-like": local_voice_like_provider,
}
both_providers = pytest.mark.parametrize("make_provider", PROVIDERS.values(), ids=PROVIDERS)

_HANG_GUARD_S = 30.0
_LOOP_CALLS = 8


def _config():
    return SimpleNamespace(
        brain=SimpleNamespace(reply_language="en", providers={}, realtime=None),
        voice=SimpleNamespace(realtime_tool_declaration_budget_tokens=0),
    )


def _tools(tmp_path, gateway: FakeToolGateway) -> LiveTools:
    ledger = LiveLedger(tmp_path / "identical-loop.sqlite3")
    return LiveTools(gateway, ledger, "identical-loop", language="en", backend_model="")


@pytest.mark.asyncio
async def test_identical_calls_without_breaker_reexecute_every_time(
    tmp_path, monkeypatch,
) -> None:
    """The loop shape without the repair: N identical calls → N gateway runs.

    This is the red baseline the breaker must turn green. Monkeypatching the
    replay hook away restores the 2026-11-14 night behaviour.
    """
    monkeypatch.setattr(
        LiveTools, "_replay_identical_receipt", lambda self, *a, **k: None,
    )
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    try:
        for i in range(_LOOP_CALLS):
            await tools.execute(f"c{i}", "search_web", args, 1)
        assert len(gateway.calls) == _LOOP_CALLS
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_identical_calls_in_one_revision_execute_once(tmp_path) -> None:
    """Offline repro of the night loop: N identical calls, one gateway execution.

    Without the identical-call breaker this assertion fails (gateway sees N) —
    see ``test_identical_calls_without_breaker_reexecute_every_time``.
    With it, only the first call runs and later receipts carry the speak-now cue.
    """
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    results = [
        await tools.execute(f"c{i}", "search_web", args, 1)
        for i in range(_LOOP_CALLS)
    ]
    try:
        assert len(gateway.calls) == 1, (
            f"identical calls must not re-execute; gateway saw {len(gateway.calls)}"
        )
        assert results[0].get("success") is True
        assert results[0].get("repeat_of_identical_call") is not True
        for later in results[1:]:
            assert later.get("reused_receipt") is True
            assert later.get("repeat_of_identical_call") is True
            assert _REPEAT_SPEAK_NOW in str(later.get("next_step") or later.get("notice") or "")
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_progressing_args_still_execute(tmp_path) -> None:
    """Different arguments are progress: each must reach the gateway."""
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    try:
        await tools.execute("c1", "search_web", {"query": "one"}, 1)
        await tools.execute("c2", "search_web", {"query": "two"}, 1)
        await tools.execute("c3", "search_web", {"query": "three"}, 1)
        assert len(gateway.calls) == 3
        assert [call[1]["query"] for call in gateway.calls] == ["one", "two", "three"]
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_new_revision_may_repeat_the_same_call(tmp_path) -> None:
    """A new user turn clears the cache; the same args may run again."""
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    args = {"query": "same"}
    try:
        tools.revision = 1
        await tools.execute("c1", "search_web", args, 1)
        tools.revision = 2
        await tools.execute("c2", "search_web", args, 2)
        assert len(gateway.calls) == 2
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_screen_tools_may_repeat_the_same_steps(tmp_path) -> None:
    """Computer rounds re-issue the same screenshot shape; do not collapse them."""
    from jarvis.core.protocols import SupervisorToolDescriptor

    gateway = FakeToolGateway(
        (
            SupervisorToolDescriptor(
                "computer",
                "Operate the screen.",
                {"type": "object", "properties": {"steps": {"type": "array"}}},
                "safe",
            ),
        )
    )
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    steps = {"steps": [{"action": "screenshot"}]}
    try:
        await tools.execute("c1", "computer", steps, 1)
        await tools.execute("c2", "computer", steps, 1)
        assert len(gateway.calls) == 2
    finally:
        tools.ledger.close()


def _in_call(provider, tmp_path, gateway):
    from tests.fakes.fake_native_voice import FakeNativeConnection

    async def send_json(_frame):
        return None

    async def send_binary(_data):
        return None

    session = NativeLiveVoiceSession(
        session_id="native-identical-loop",
        send_json=send_json,
        send_binary=send_binary,
        providers=[provider],
        config=_config(),
    )
    connection = FakeNativeConnection(
        creates_responses_automatically=provider._creates, model="m",
    )
    ledger = LiveLedger(tmp_path / f"{provider.name}-identical.sqlite3")
    session._connection = connection
    session._ledger = ledger
    session._tools = LiveTools(
        gateway, ledger, "native-identical-loop", language="en", backend_model="",
    )
    session._tools.revision = 1
    return session, connection


@both_providers
@pytest.mark.asyncio
async def test_native_turn_stops_identical_tool_loop(
    tmp_path, monkeypatch, make_provider
) -> None:
    """A real native turn: N identical tool_call events, one gateway execution."""
    monkeypatch.setattr(native, "_TOOL_DEADLINE_S", _HANG_GUARD_S)
    gateway = FakeToolGateway()
    session, connection = _in_call(make_provider(), tmp_path, gateway)
    try:
        for i in range(_LOOP_CALLS):
            connection.tool_result_sent.clear()
            await session._native_event(
                RealtimeEvent(
                    type="tool_call",
                    call_id=f"loop-{i}",
                    tool_name="search_web",
                    tool_args={"query": "memory question"},
                )
            )
            await asyncio.wait_for(connection.tool_result_sent.wait(), _HANG_GUARD_S)

        assert len(gateway.calls) == 1, (
            f"native identical loop must stop; gateway saw {len(gateway.calls)}"
        )
        assert len(connection.tool_results) == _LOOP_CALLS
        first = connection.tool_results[0][2]
        assert first.get("success") is True
        for _call_id, _name, result in connection.tool_results[1:]:
            assert result.get("repeat_of_identical_call") is True
            assert _REPEAT_SPEAK_NOW in str(
                result.get("next_step") or result.get("notice") or ""
            )
    finally:
        session._ledger.close()


class _NeverRemembers(dict):
    """A registry that forgets: restores the lane-free concurrent loop shape."""

    def get(self, _key, default=None):  # noqa: D102 - see class docstring
        return default


@pytest.mark.asyncio
async def test_concurrent_identical_calls_without_guard_all_execute(tmp_path) -> None:
    """Red baseline for the burst: ``_lane`` does not serialise these tools.

    ``workspace-orchestrate`` and ``run_app_action`` — the two tools in the
    2026-11-14 storm — are neither screen nor control tools, so ``_lane``
    returns ``None`` and identical calls run side by side. The finished-receipt
    cache cannot see them: at 2-3 calls/s none of them has recorded a receipt
    yet. Without the in-flight registry every call reaches the gateway.
    """
    gateway = FakeToolGateway()
    gate = gateway.hold("search_web")
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    tools._inflight_by_identity = _NeverRemembers()
    args = {"query": "favourite dish"}
    try:
        burst = [
            asyncio.create_task(tools.execute(f"c{i}", "search_web", args, 1))
            for i in range(_LOOP_CALLS)
        ]
        await asyncio.wait_for(gateway.execution_started.wait(), _HANG_GUARD_S)
        await asyncio.sleep(0)
        gate.set()
        await asyncio.wait_for(asyncio.gather(*burst), _HANG_GUARD_S)
        assert len(gateway.calls) == _LOOP_CALLS
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_concurrent_identical_calls_execute_once(tmp_path) -> None:
    """The burst repaired: N overlapping identical calls, one gateway execution.

    Each repeat awaits the running call and answers from its receipt, so every
    call still gets a real result and the model is told to speak.
    """
    gateway = FakeToolGateway()
    gate = gateway.hold("search_web")
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    try:
        burst = [
            asyncio.create_task(tools.execute(f"c{i}", "search_web", args, 1))
            for i in range(_LOOP_CALLS)
        ]
        await asyncio.wait_for(gateway.execution_started.wait(), _HANG_GUARD_S)
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*burst), _HANG_GUARD_S)
        assert len(gateway.calls) == 1, (
            f"overlapping identical calls must run once; gateway saw {len(gateway.calls)}"
        )
        repeats = [r for r in results if r.get("repeat_of_identical_call") is True]
        assert len(repeats) == _LOOP_CALLS - 1
        for repeat in repeats:
            assert repeat.get("success") is True
            assert _REPEAT_SPEAK_NOW in str(
                repeat.get("next_step") or repeat.get("notice") or ""
            )
    finally:
        tools.ledger.close()
