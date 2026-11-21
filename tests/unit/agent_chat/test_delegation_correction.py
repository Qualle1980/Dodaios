"""One bounded correction attempt after a rejected handback (Phase C2).

The acceptance gate rejects an answer whose evidence is missing. Slice 1 wrote
that down; this is the return path. The whole risk of it lives in one
distinction: a job that DELIVERED and cannot prove it may be corrected, while a
job that is blocked, crashed or superseded may not be re-dispatched on
suspicion -- ``test_a_blocked_or_failed_agent_is_never_accepted_and_never_retried``
is that contract, and these tests pin the boundary it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.agent_chat.delegation_correction import (
    MAX_CORRECTIONS,
    correctable,
    correction_prompt,
)

ORDER = "Add subtract() to the calculator and run the tests."
GAPS = ("no test run inside the job's checkout", "no change read")


@dataclass(frozen=True)
class FakeJob:
    task_id: str
    status: str
    workspace_id: str = "ws-1"


@dataclass(frozen=True)
class FakeGate:
    jobs: tuple[FakeJob, ...]

    @property
    def work(self) -> tuple[str, ...]:
        return tuple(job.task_id for job in self.jobs)


def test_only_a_job_that_claims_delivery_may_be_corrected() -> None:
    """``completed`` is the one status that claims the work was handed back.
    Everything else is further from acceptance, not closer to it, so it is not
    re-dispatched: the person is told honestly instead."""
    assert correctable(FakeGate((FakeJob("coding-1", "completed"),))) == ("coding-1",)
    for status in ("needs_input", "error", "cancelled", "superseded", "running", ""):
        gate = FakeGate((FakeJob("coding-1", status),))
        assert correctable(gate) == (), status


def test_a_mixed_wave_corrects_only_the_jobs_that_delivered() -> None:
    gate = FakeGate((
        FakeJob("coding-1", "completed"),
        FakeJob("coding-2", "needs_input"),
        FakeJob("coding-3", "completed"),
    ))
    assert correctable(gate) == ("coding-1", "coding-3")


def test_a_gate_without_jobs_is_not_correctable() -> None:
    assert correctable(FakeGate(())) == ()
    assert correctable(None) == ()


def test_the_correction_names_the_gaps_and_forbids_a_claim() -> None:
    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules="", attempt=1
    )
    for gap in GAPS:
        assert gap in prompt
    assert "coding-1" in prompt
    assert ORDER in prompt
    # The correction is an instruction to get evidence, never a licence to
    # claim the work is good.
    lowered = prompt.lower()
    assert "do not claim" in lowered or "never claim" in lowered
    assert "attempt 1" in lowered and str(MAX_CORRECTIONS) in prompt


def test_the_learned_rules_travel_with_the_correction() -> None:
    """The conductor's seat may have written a rule from the first failure
    (slice 1). The next attempt has to see it, or the review bought nothing."""
    rules = "- Run the tests inside the job's own checkout before claiming a pass."
    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules=rules, attempt=1
    )
    assert rules in prompt
    assert "attempt 1" in prompt.lower()


def test_the_correction_is_not_read_as_the_persons_own_words() -> None:
    """A composed prompt that quotes the order would otherwise be routed as if
    the person had typed it, which is how a pane got re-ordered once
    (``is_continuation_prompt``)."""
    from jarvis.agent_chat.delegation_wait import is_continuation_prompt

    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules="", attempt=1
    )
    assert is_continuation_prompt(prompt)


def test_one_attempt_is_the_bound() -> None:
    """A real coding run costs money per round, so the bound is small and
    explicit rather than left to the wave budget. One Instant correction;
    a second rejection ends the request honestly."""
    assert MAX_CORRECTIONS == 1


def test_correction_prompt_carries_one_line_status_per_job() -> None:
    """The correction briefing names each job's status in one line; it does
    not replay earlier in-turn tool dumps."""
    prompt = correction_prompt(
        gaps=GAPS,
        work=("coding-1", "coding-2"),
        original=ORDER,
        rules="",
        attempt=1,
        statuses=(("coding-1", "completed"), ("coding-2", "needs_input")),
    )
    assert "coding-1: completed" in prompt
    assert "coding-2: needs_input" in prompt


def test_second_rejection_ends_honestly_without_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one Instant correction, a second rejection ends the request: it
    names the open gap and does not start a third attempt."""
    import asyncio
    from types import SimpleNamespace
    from typing import Any

    from jarvis.agent_chat import delegation_wait
    from jarvis.agent_chat.delegation_acceptance import DIFF_MISSING, TEST_MISSING
    from jarvis.agent_chat.events import make_event
    from jarvis.agent_chat.runner_api import TurnHandle
    from jarvis.agent_chat.store import AgentChatStore
    from jarvis.core.protocols import ChatTurn, current_chat_turn

    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(delegation_wait, "WAIT_TIMEOUT_SECONDS", 1.0)

    async def no_codex(*_a: Any, **_k: Any) -> bool:
        return False

    monkeypatch.setattr(delegation_wait, "_maybe_codex_fallback", no_codex)

    events: list[dict[str, Any]] = []
    histories: list[list[dict[str, Any]]] = []
    prompts: list[str] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    async def approve(*_a: Any, **_k: Any) -> str:
        return "allow"

    store = AgentChatStore(":memory:")
    session = store.create_session(
        provider="openai", model="", effort="", surface="jarvis", cwd="."
    )
    # Seed a prior tool dump so a naive history replay would carry it.
    store.append_event(
        session.session_id,
        make_event(
            "tool_result",
            {
                "turn_id": "t1",
                "call_id": "c1",
                "name": "Read",
                "output": "x" * 5000,
                "is_error": False,
            },
        ),
    )
    handle = TurnHandle(
        session=session,
        turn_id="t1",
        emit=emit,
        request_approval=approve,
        cancel=asyncio.Event(),
        output_language="en",
        acceptance_gate=FakeGate((FakeJob("coding-1", "completed"),)),
        corrections_left=MAX_CORRECTIONS,
    )
    service = SimpleNamespace(store=store, delegate_wait=None)
    waves = {"n": 0}

    async def run_attempt(active: Any, prompt: str) -> None:
        waves["n"] += 1
        prompts.append(prompt)
        histories.append(list(active.history or []))
        await active.emit(
            make_event(
                "notice",
                {
                    "kind": "delegation_correction",
                    "turn_id": active.turn_id,
                    "work": ["coding-1"],
                    "missing": [DIFF_MISSING, TEST_MISSING],
                    "correcting": ["coding-1"],
                    "left": active.corrections_left,
                    "receipts": [],
                    "verdict_text": "",
                    "tool_error": False,
                },
            )
        )
        await active.emit(
            make_event(
                "turn_finished",
                {"turn_id": active.turn_id, "status": "done", "usage": {}},
            )
        )

    token = current_chat_turn.set(
        ChatTurn(session.session_id, "t1", ORDER, True, "trace")
    )
    try:

        async def run() -> None:
            await delegation_wait.run_with_delegates(service, handle, ORDER, run_attempt)

        asyncio.run(run())
    finally:
        current_chat_turn.reset(token)
        store.close()

    # Initial attempt + one correction only — never a third try.
    assert waves["n"] == 2
    assert MAX_CORRECTIONS == 1
    # Correction attempt must not replay earlier in-turn tool dumps.
    assert histories[1] == []
    assert "coding-1: completed" in prompts[1]
    # Honest end: open gaps named, no third attempt.
    texts = [
        str((e.get("payload") or {}).get("text") or "")
        for e in events
        if e.get("kind") == "assistant_text"
    ]
    honest = "\n".join(texts).lower()
    assert honest, "second rejection must speak the honest line"
    assert "unread" in honest or "change" in honest
    assert "test" in honest
