"""A tool lead that is still setting itself up must not hold the turn.

Measured live on 2026-11-06. A voice turn on a talker that cannot call tools:
the seat policy left only a keyless local runtime to lead, that runtime had no
model configured, and it chose one and built a profile INSIDE its first call.
The user heard the timeout filler phrase 30 s later and the turn record kept an
empty answer and no provider.

Neither existing bound could fire. ``ToolUseLoop.deadline_s`` is checked
between tool rounds and the first round never finished; the stream stall
watchdog waits 30 s, which is right for an answer and far too long for a
provider whose only job is to name a tool.

These tests pin the halves of the repair:

1. A lead that shows no sign of life inside the budget is abandoned.
2. A sign of life spends the budget for good — the work itself is never cut.
3. Abandoning is OUR deadline, so it carries no verdict about the provider.
"""
from __future__ import annotations

import asyncio

import pytest

from jarvis.brain.lead_budget import (
    LEAD_FIRST_SIGN_BUDGET_S,
    LeadActivity,
    LeadTookTooLong,
    run_lead_under_budget,
)


async def _run(attempt, activity, *, budget_s: float = 0.2):
    return await run_lead_under_budget(
        attempt, activity, provider="fake", model="m", budget_s=budget_s, poll_s=0.01,
    )


# ── the activity flag ────────────────────────────────────────────────────────


def test_a_fresh_lead_has_shown_nothing() -> None:
    assert LeadActivity().seen is False


def test_any_sign_counts_once_and_stays() -> None:
    activity = LeadActivity()

    activity.mark()
    activity.mark()

    assert activity.seen is True


def test_a_wrapped_callback_still_runs_and_still_returns() -> None:
    activity = LeadActivity()
    seen: list[tuple[str, int]] = []

    wrapped = activity.wrap(lambda text, n: seen.append((text, n)) or "passed through")
    result = wrapped("chunk", 7)

    assert seen == [("chunk", 7)]
    assert result == "passed through"
    assert activity.seen is True


def test_a_wrapped_none_callback_still_marks() -> None:
    """The lead's text consumer is deliberately ``None``; the signal still counts."""
    activity = LeadActivity()

    assert activity.wrap(None)("anything") is None
    assert activity.seen is True


# ── the budget ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_lead_that_shows_nothing_is_abandoned() -> None:
    """The shape of the live failure: setup inside the first call, no output."""
    async def _silent_setup() -> str:
        await asyncio.sleep(30)
        return "never reached"

    with pytest.raises(LeadTookTooLong) as caught:
        await _run(_silent_setup, LeadActivity())

    assert caught.value.provider == "fake"
    assert "without a tool lead" in str(caught.value)


@pytest.mark.asyncio
async def test_a_lead_that_answers_inside_the_budget_is_returned() -> None:
    async def _quick() -> str:
        await asyncio.sleep(0.01)
        return "picked a tool"

    assert await _run(_quick, LeadActivity()) == "picked a tool"


@pytest.mark.asyncio
async def test_a_sign_of_life_spends_the_budget_so_slow_work_is_not_cut() -> None:
    """The reason this is a FIRST-SIGN budget: a slow tool keeps its time."""
    activity = LeadActivity()

    async def _works_then_takes_its_time() -> str:
        activity.mark()          # a progress note, an ack, a chunk
        await asyncio.sleep(0.5)  # well past the 0.2 s budget
        return "slow tool, real answer"

    assert await _run(_works_then_takes_its_time, activity) == "slow tool, real answer"


@pytest.mark.asyncio
async def test_the_abandoned_attempt_is_actually_cancelled() -> None:
    """A lead left running would keep spending on a turn nobody waits for."""
    cancelled = asyncio.Event()

    async def _silent_setup() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never reached"

    with pytest.raises(LeadTookTooLong):
        await _run(_silent_setup, LeadActivity())

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_the_lead_s_own_error_is_not_swallowed_when_it_is_in_time() -> None:
    """Only OUR deadline is special; a real failure still reaches the chain."""
    async def _broken() -> str:
        raise RuntimeError("401 Unauthorized")

    with pytest.raises(RuntimeError, match="401"):
        await _run(_broken, LeadActivity())


@pytest.mark.asyncio
async def test_a_lead_that_fails_silently_past_the_budget_blames_our_deadline() -> None:
    """Its error arrived too late to matter and must not read as the cause."""
    async def _fails_after_a_while() -> str:
        await asyncio.sleep(30)
        raise RuntimeError("would have been a provider error")

    with pytest.raises(LeadTookTooLong):
        await _run(_fails_after_a_while, LeadActivity())


def test_the_budget_sits_between_a_healthy_lead_and_the_stall_window() -> None:
    """Measured 2026-11-06: 0.6 s gemini-flash-lite, 2.1 s chatgpt-plan,
    2.7 s to a healthy lead's first tool proposal, 26 s for a local runtime
    that had to build a model profile. The stall watchdog is 30 s."""
    assert 2.7 < LEAD_FIRST_SIGN_BUDGET_S < 26.0
