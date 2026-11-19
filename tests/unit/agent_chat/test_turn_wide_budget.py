"""Turn-wide IterationBudget across DelegationWait waves (slice 1, 2026-11-17)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agent_chat import delegation_wait
from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.runner_api import TurnHandle
from jarvis.agent_chat.store import AgentChatStore
from jarvis.brain.iteration_budget import IterationBudget
from jarvis.core.delegated_work import DelegatedWork, register_delegated_work
from jarvis.core.protocols import ChatTurn, current_chat_turn

ORDER = "Create hallo.txt with hallo, then independent review."


@pytest.mark.asyncio
async def test_waves_share_one_budget_and_stop_when_input_exceeds_explicit_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waves accumulate input on one budget; the next wave stops past an explicit ceiling.

    The shipped default input backstop is 50M (mandate 2026-08-24). This pins the
    turn-wide stop itself with an operator-set 80k ceiling on the handle.
    """
    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(delegation_wait, "WAIT_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(delegation_wait, "MAX_CORRECTIONS", 0)

    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    async def approve(*_a: Any, **_k: Any) -> str:
        return "allow"

    store = AgentChatStore(":memory:")
    session = store.create_session(
        provider="chatgpt-plan", model="gpt-x", effort="", surface="jarvis", cwd="."
    )
    ceiling = IterationBudget(max_turns=40, max_input_tokens_total=80_000)
    handle = TurnHandle(
        session=session,
        turn_id="t-budget",
        emit=emit,
        request_approval=approve,
        cancel=asyncio.Event(),
        output_language="en",
        iteration_budget=ceiling,
    )
    service = SimpleNamespace(store=store, delegate_wait=None)
    seen: list[IterationBudget] = []
    waves = {"n": 0}

    async def run_attempt(active: Any, prompt: str) -> None:
        waves["n"] += 1
        budget = getattr(active, "iteration_budget", None)
        assert isinstance(budget, IterationBudget), "each wave must carry the turn budget"
        seen.append(budget)
        # Wave 1 under the cap; wave 2 crosses 80k; wave 3 must not run.
        if waves["n"] == 1:
            budget.record_turn(tokens_in=50_000, tokens_out=10)
            async def probe() -> dict[str, Any]:
                return {"status": "completed", "report": "file written"}

            assert register_delegated_work(DelegatedWork("coding:1", "Claude Code", probe))
            used = 50_000
        else:
            budget.record_turn(tokens_in=40_000, tokens_out=10)
            used = 40_000
            if waves["n"] == 2:
                async def probe2() -> dict[str, Any]:
                    return {"status": "completed", "report": "review done"}

                assert register_delegated_work(DelegatedWork("coding:2", "Codex", probe2))
        await active.emit(
            make_event(
                "turn_finished",
                {
                    "turn_id": active.turn_id,
                    "status": "done",
                    "usage": {"input_tokens": used},
                },
            )
        )

    token = current_chat_turn.set(
        ChatTurn(session.session_id, "t-budget", ORDER, True, "trace")
    )
    try:
        await delegation_wait.run_with_delegates(service, handle, ORDER, run_attempt)
    finally:
        current_chat_turn.reset(token)
        store.close()

    assert waves["n"] == 2, "wave 3 must not start after cumulative input crossed the ceiling"
    assert seen[0] is seen[1]
    assert seen[0].input_tokens_seen >= 80_000
    assert seen[0] is ceiling


@pytest.mark.asyncio
async def test_under_cap_runs_all_waves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn under the input limit still runs every needed wave on one budget."""
    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(delegation_wait, "WAIT_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(delegation_wait, "MAX_CORRECTIONS", 0)

    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    async def approve(*_a: Any, **_k: Any) -> str:
        return "allow"

    store = AgentChatStore(":memory:")
    session = store.create_session(
        provider="chatgpt-plan", model="gpt-x", effort="", surface="jarvis", cwd="."
    )
    handle = TurnHandle(
        session=session,
        turn_id="t-under",
        emit=emit,
        request_approval=approve,
        cancel=asyncio.Event(),
        output_language="en",
    )
    service = SimpleNamespace(store=store, delegate_wait=None)
    budgets: list[IterationBudget] = []
    waves = {"n": 0}

    async def run_attempt(active: Any, prompt: str) -> None:
        waves["n"] += 1
        budget = getattr(active, "iteration_budget", None)
        assert isinstance(budget, IterationBudget)
        budgets.append(budget)
        budget.record_turn(tokens_in=20_000, tokens_out=5)
        if waves["n"] == 1:
            async def probe() -> dict[str, Any]:
                return {"status": "completed", "report": "ok"}

            assert register_delegated_work(DelegatedWork("coding:1", "Claude Code", probe))
        await active.emit(
            make_event(
                "turn_finished",
                {
                    "turn_id": active.turn_id,
                    "status": "done",
                    "usage": {"input_tokens": 20_000},
                },
            )
        )

    token = current_chat_turn.set(
        ChatTurn(session.session_id, "t-under", ORDER, True, "trace")
    )
    try:
        await delegation_wait.run_with_delegates(service, handle, ORDER, run_attempt)
    finally:
        current_chat_turn.reset(token)
        store.close()

    assert waves["n"] == 2
    assert budgets[0] is budgets[1], "waves must share one IterationBudget object"
    assert budgets[0].input_tokens_seen == 40_000


def test_dispatcher_reuses_override_budget() -> None:
    """BrainDispatcher must not allocate a fresh budget when the turn passed one."""
    from collections.abc import AsyncIterator

    from jarvis.brain.dispatcher import BrainDispatcher
    from jarvis.core.protocols import BrainDelta, BrainRequest, ToolResult

    class _Brain:
        async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
            yield BrainDelta(content="done")
            yield BrainDelta(finish_reason="stop", usage={"input_tokens": 100, "output_tokens": 1})

    class _Tool:
        name = "noop"
        schema: dict[str, Any] = {}

    class _Exec:
        async def execute(self, tool: Any, args: dict[str, Any], **_: Any) -> ToolResult:
            return ToolResult(success=True, output="ok")

    shared = IterationBudget(max_turns=6, max_input_tokens_total=80_000)
    shared.record_turn(tokens_in=50_000, tokens_out=1)
    disp = BrainDispatcher(
        _Brain(),  # type: ignore[arg-type]
        tools={"noop": _Tool()},  # type: ignore[dict-item]
        executor=_Exec(),  # type: ignore[arg-type]
        budget=shared,
        max_turns=6,
    )

    async def run() -> None:
        await disp.dispatch("hello")

    asyncio.run(run())
    assert shared.input_tokens_seen == 50_100
    assert shared.turns_used >= 1
