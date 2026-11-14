"""Identical live tool calls in one turn must stop after a small sequential cap.

2026-11-14 night sessions: the model called ``workspace-orchestrate`` (filter
on) or ``run_app_action`` (filter off) dozens of times at 2–3 calls/s with
``turn complete (function call, boundary withheld) — audio=0.0s``. Most
generations never spoke. Flight recorder showed many identical
``{"action": "status"}`` successes and opaque call-id collisions.

PR #142 fixed the empty-schema memory-tool cause of the 2026-11-09 177-call
loop but deliberately deferred a per-turn identical-call bound. Round 1 of
this PR answered every repeat with a cached receipt plus speak-now; a
non-compliant model kept looping faster. This file is the bound that ends
the round: reads may replay once, effectful args re-run a few times, then
further identical calls withhold the function response.

These tests drive real ``LiveTools`` / ``NativeLiveVoiceSession`` over fakes
from ``tests/fakes/`` — they execute calls, they do not only assert the catalog.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from jarvis.core.protocols import SupervisorToolDescriptor, ToolResult
from jarvis.live import native
from jarvis.live.native import NativeLiveVoiceSession
from jarvis.live.state import LiveLedger
from jarvis.live.tools import (
    LiveTools,
    _APPROVAL_NEXT_STEP,
    _IDENTICAL_EFFECT_SEQUENTIAL_CAP,
    _IDENTICAL_READ_SEQUENTIAL_CAP,
    _REPEAT_SPEAK_NOW,
    _REPEAT_STORM_ENDED,
)
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


def _action_gateway() -> FakeToolGateway:
    return FakeToolGateway(
        (
            SupervisorToolDescriptor(
                "run_app_action",
                "Run an app action.",
                {"type": "object", "properties": {"action": {"type": "string"}}},
                "monitor",
                is_action_tool=True,
            ),
        )
    )


@pytest.mark.asyncio
async def test_identical_calls_without_breaker_reexecute_every_time(
    tmp_path, monkeypatch,
) -> None:
    """The loop shape without the repair: N identical calls → N gateway runs.

    This is the red baseline the breaker must turn green. Disabling identity
    restores the 2026-11-14 night behaviour.
    """
    monkeypatch.setattr(LiveTools, "_tool_call_identity", lambda self, *a, **k: None)
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
async def test_identical_read_replays_once_then_withholds(tmp_path) -> None:
    """Read: first executes, second replays speak-now, further calls withhold."""
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
            f"identical reads must not re-execute; gateway saw {len(gateway.calls)}"
        )
        assert results[0].get("success") is True
        assert results[0].get("repeat_of_identical_call") is not True
        assert results[0].get("withhold_tool_response") is not True
        # Wave 2: advisory replay
        assert results[1].get("reused_receipt") is True
        assert results[1].get("repeat_of_identical_call") is True
        assert _REPEAT_SPEAK_NOW in str(results[1].get("next_step") or results[1].get("notice") or "")
        assert results[1].get("withhold_tool_response") is not True
        # Waves 3..N: end the round
        for later in results[_IDENTICAL_READ_SEQUENTIAL_CAP:]:
            assert later.get("withhold_tool_response") is True
            assert later.get("end_tool_round") is True
            assert _REPEAT_STORM_ENDED in str(later.get("error") or "")
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


async def _drain_jobs(session: NativeLiveVoiceSession) -> None:
    pending = [job for job in list(session._jobs) if not job.done()]
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), _HANG_GUARD_S)


@both_providers
@pytest.mark.asyncio
async def test_native_turn_stops_identical_tool_loop(
    tmp_path, monkeypatch, make_provider
) -> None:
    """Native path: answer up to the read cap, then withhold further responses."""
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
            await _drain_jobs(session)

        assert len(gateway.calls) == 1, (
            f"native identical loop must stop; gateway saw {len(gateway.calls)}"
        )
        assert len(connection.tool_results) == _IDENTICAL_READ_SEQUENTIAL_CAP
        first = connection.tool_results[0][2]
        assert first.get("success") is True
        second = connection.tool_results[1][2]
        assert second.get("repeat_of_identical_call") is True
        assert _REPEAT_SPEAK_NOW in str(
            second.get("next_step") or second.get("notice") or ""
        )
    finally:
        session._ledger.close()


class _NeverRemembers(dict):
    """A registry that forgets: restores the lane-free concurrent loop shape."""

    def get(self, _key, default=None):  # noqa: D102 - see class docstring
        return default


@pytest.mark.asyncio
async def test_concurrent_identical_calls_without_guard_all_execute(tmp_path) -> None:
    """Red baseline for the burst: ``_lane`` does not serialise these tools."""
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
    """Overlapping identical reads: one gateway run; waiters share that wave."""
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
        assert all(r.get("withhold_tool_response") is not True for r in results)
        repeats = [r for r in results if r.get("repeat_of_identical_call") is True]
        assert len(repeats) == _LOOP_CALLS - 1
        for repeat in repeats:
            assert repeat.get("success") is True
            assert _REPEAT_SPEAK_NOW in str(
                repeat.get("next_step") or repeat.get("notice") or ""
            )
    finally:
        tools.ledger.close()


# --- Review findings F1–F7, F9, F10 -----------------------------------------


@pytest.mark.asyncio
async def test_f1_identical_waiter_gets_receipt_when_primary_cancelled(tmp_path) -> None:
    """F1: a waiter must not inherit the primary's CancelledError."""
    gateway = FakeToolGateway()
    gate = gateway.hold("search_web")
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    try:
        primary = asyncio.create_task(tools.execute("c0", "search_web", args, 1))
        await asyncio.wait_for(gateway.execution_started.wait(), _HANG_GUARD_S)
        waiter = asyncio.create_task(tools.execute("c1", "search_web", args, 1))
        await asyncio.sleep(0)
        primary.cancel()
        with pytest.raises(asyncio.CancelledError):
            await primary
        result = await asyncio.wait_for(waiter, _HANG_GUARD_S)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert result.get("repeat_of_identical_call") is True
    finally:
        gate.set()
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f2_failure_receipt_is_not_cached(tmp_path) -> None:
    """F2: a hard failure must not permanently block a later identical retry."""
    gateway = FakeToolGateway()
    gateway.results["search_web"] = ToolResult(False, {}, error="temporary outage")
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    try:
        first = await tools.execute("c0", "search_web", args, 1)
        assert first.get("success") is False
        gateway.results["search_web"] = ToolResult(True, {"verified": True})
        second = await tools.execute("c1", "search_web", args, 1)
        assert len(gateway.calls) == 2
        assert second.get("success") is True
        assert second.get("repeat_of_identical_call") is not True
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f3_effectful_call_invalidates_cached_read(tmp_path) -> None:
    """F3: after an action, an identical read must see fresh state."""
    gateway = FakeToolGateway(
        (
            SupervisorToolDescriptor(
                "search_web",
                "Search the web.",
                {"type": "object", "properties": {"query": {"type": "string"}}},
                "safe",
            ),
            SupervisorToolDescriptor(
                "run_app_action",
                "Run an app action.",
                {"type": "object", "properties": {"action": {"type": "string"}}},
                "monitor",
                is_action_tool=True,
            ),
        )
    )
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    read_args = {"query": "status"}
    try:
        first = await tools.execute("c0", "search_web", read_args, 1)
        assert first.get("success") is True
        await tools.execute("c1", "run_app_action", {"action": "next"}, 1)
        second = await tools.execute("c2", "search_web", read_args, 1)
        read_calls = [c for c in gateway.calls if c[0] == "search_web"]
        assert len(read_calls) == 2
        assert second.get("repeat_of_identical_call") is not True
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f4_invalidate_identical_receipts_clears_cache(tmp_path) -> None:
    """F4: application-event invalidation lets the next read run again."""
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "status"}
    try:
        await tools.execute("c0", "search_web", args, 1)
        tools.invalidate_identical_receipts(reset_waves=True)
        await tools.execute("c1", "search_web", args, 1)
        assert len(gateway.calls) == 2
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f5_effectful_identical_args_reexecute_until_cap(tmp_path) -> None:
    """F5: deliberate effectful repeats run; past the effect cap they withhold."""
    gateway = _action_gateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"action": "skip"}
    try:
        results = [
            await tools.execute(f"c{i}", "run_app_action", args, 1)
            for i in range(_IDENTICAL_EFFECT_SEQUENTIAL_CAP + 2)
        ]
        assert len(gateway.calls) == _IDENTICAL_EFFECT_SEQUENTIAL_CAP
        for early in results[:_IDENTICAL_EFFECT_SEQUENTIAL_CAP]:
            assert early.get("success") is True
            assert early.get("withhold_tool_response") is not True
        for later in results[_IDENTICAL_EFFECT_SEQUENTIAL_CAP:]:
            assert later.get("withhold_tool_response") is True
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f6_approval_next_step_survives_identical_replay(tmp_path, monkeypatch) -> None:
    """F6: replaying a gated action keeps the approval instruction."""
    gateway = _action_gateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"action": "delete"}
    gated = {
        "success": False,
        "confirmation_required": True,
        "approval_id": str(uuid4()),
        "tool": "run_app_action",
        "arguments": args,
        "next_step": _APPROVAL_NEXT_STEP,
    }

    async def _fake_claim_and_run(
        self, call_id, name, args, revision, replay_result, *, identity=None,
    ):
        del call_id, replay_result
        self._remember_identical_receipt(
            name, args, revision, gated, identity=identity,
        )
        return gated, dict(gated)

    monkeypatch.setattr(LiveTools, "_claim_and_run", _fake_claim_and_run)
    try:
        first = await tools.execute("c0", "run_app_action", args, 1)
        assert first.get("confirmation_required") is True
        second = await tools.execute("c1", "run_app_action", args, 1)
        assert second.get("repeat_of_identical_call") is True
        assert second.get("next_step") == _APPROVAL_NEXT_STEP
        assert _APPROVAL_NEXT_STEP in str(second.get("next_step") or "")
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f7_identical_waiter_reuses_released_pending_receipt(tmp_path) -> None:
    """F7: if the primary is already backgrounded, waiters get its task_id."""
    gateway = FakeToolGateway()
    gate = gateway.hold("search_web")
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "favourite dish"}
    try:
        primary = asyncio.create_task(
            tools.execute_releasable("c0", "search_web", args, 1, deadline_s=0.05)
        )
        await asyncio.wait_for(gateway.execution_started.wait(), _HANG_GUARD_S)
        # Let the deadline release the primary to the background.
        pending = await asyncio.wait_for(primary, _HANG_GUARD_S)
        assert pending.get("pending") is True
        task_id = pending.get("task_id")
        assert task_id
        waiter = await tools.execute("c1", "search_web", args, 1)
        assert waiter.get("pending") is True
        assert waiter.get("task_id") == task_id
        assert waiter.get("repeat_of_identical_call") is True
        assert len(tools.tasks.running()) == 1
    finally:
        gate.set()
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f9_tool_call_identity_computed_once_per_execution(tmp_path, monkeypatch) -> None:
    """F9: identity is computed once in ``_execute_recorded`` and passed down."""
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    calls = {"n": 0}
    original = LiveTools._tool_call_identity

    def counted(self, name: str, args: dict) -> str | None:
        calls["n"] += 1
        return original(self, name, args)

    monkeypatch.setattr(LiveTools, "_tool_call_identity", counted)
    try:
        await tools.execute("c0", "search_web", {"query": "once"}, 1)
        # One computation for the hot path; remember receives the passed identity.
        assert calls["n"] == 1
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f10_without_sequential_cap_answers_every_repeat(tmp_path, monkeypatch) -> None:
    """Red baseline for F10: without a sequential cap every repeat is answered.

    That is the round-1 shape the cloud review rejected: advisory speak-now
    forever, which a non-compliant model turns into a faster paid loop.
    """
    monkeypatch.setattr(
        LiveTools, "_identical_sequential_cap", lambda self, name, args: 10_000,
    )
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "loop"}
    try:
        results = [
            await tools.execute(f"c{i}", "search_web", args, 1)
            for i in range(_LOOP_CALLS)
        ]
        assert len(gateway.calls) == 1
        assert all(r.get("withhold_tool_response") is not True for r in results)
        assert len(results) == _LOOP_CALLS
    finally:
        tools.ledger.close()


@pytest.mark.asyncio
async def test_f10_sequential_identical_reads_withhold_after_cap(tmp_path) -> None:
    """F10: past the read cap the receipt asks the session to withhold."""
    gateway = FakeToolGateway()
    tools = _tools(tmp_path, gateway)
    tools.revision = 1
    args = {"query": "loop"}
    try:
        results = [
            await tools.execute(f"c{i}", "search_web", args, 1)
            for i in range(_IDENTICAL_READ_SEQUENTIAL_CAP + 1)
        ]
        assert results[-1].get("withhold_tool_response") is True
        assert results[-1].get("end_tool_round") is True
        assert len(gateway.calls) == 1
    finally:
        tools.ledger.close()


@both_providers
@pytest.mark.asyncio
async def test_f10_native_withholds_after_read_cap(
    tmp_path, monkeypatch, make_provider
) -> None:
    """F10 on the wire: tool_results stop after the read sequential cap."""
    monkeypatch.setattr(native, "_TOOL_DEADLINE_S", _HANG_GUARD_S)
    gateway = FakeToolGateway()
    session, connection = _in_call(make_provider(), tmp_path, gateway)
    try:
        for i in range(_IDENTICAL_READ_SEQUENTIAL_CAP + 3):
            await session._native_event(
                RealtimeEvent(
                    type="tool_call",
                    call_id=f"storm-{i}",
                    tool_name="search_web",
                    tool_args={"query": "memory question"},
                )
            )
            await _drain_jobs(session)
        assert len(connection.tool_results) == _IDENTICAL_READ_SEQUENTIAL_CAP
        assert len(gateway.calls) == 1
    finally:
        session._ledger.close()
