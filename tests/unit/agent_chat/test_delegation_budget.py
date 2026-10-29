"""What one request may spend on coding agents, and when the answer is "unknown"."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.agent_chat.delegation_budget import (
    MAX_CODING_DELEGATIONS,
    budget_spent,
    coding_delegations,
    may_delegate,
    quota_verdict,
)
from jarvis.agent_usage import AccountUsage, UsageWindow


def _window(used: float, *, kind: str = "session") -> UsageWindow:
    return UsageWindow(kind=kind, percent=used, severity="normal")


def _usage(*windows: UsageWindow, status: str = "ok", source: str = "live") -> AccountUsage:
    return AccountUsage(
        account_id="a1", platform="claude", status=status, windows=windows, source=source
    )


# ------------------------------------------------------- how many delegations


def test_only_coding_work_counts_against_the_delegation_budget() -> None:
    """A browser read or a dispatch reservation is not a coding seat."""
    keys = ["coding:t1:1:2", "dispatch:4711", "coding:t2:1:3", "mission:x"]
    assert coding_delegations(keys) == 2


def test_the_same_pane_twice_is_one_delegation() -> None:
    assert coding_delegations(["coding:t1:1:2", "coding:t1:1:2"]) == 1


def test_an_empty_turn_has_spent_nothing() -> None:
    assert coding_delegations(None) == 0
    assert not budget_spent([])


def test_the_budget_is_spent_at_the_ceiling_not_after_it() -> None:
    keys = [f"coding:t{n}:1:1" for n in range(MAX_CODING_DELEGATIONS)]
    assert budget_spent(keys)
    assert not budget_spent(keys[:-1])


# ------------------------------------------------------------- quota readings


def test_a_full_window_allows_new_work() -> None:
    verdict = quota_verdict(_usage(_window(12.0)))
    assert verdict.state == "allowed" and not verdict.blocked


@pytest.mark.parametrize("used", [90.0, 95.0, 100.0, 103.0])
def test_ten_percent_or_less_left_blocks_new_work(used) -> None:
    verdict = quota_verdict(_usage(_window(min(used, 100.0))))
    assert verdict.blocked
    assert "no new coding agent was started" in verdict.reason


def test_the_tightest_window_decides() -> None:
    """A fresh 5-hour window does not license spending an exhausted weekly one."""
    verdict = quota_verdict(_usage(_window(10.0), _window(97.0, kind="weekly")))
    assert verdict.blocked and "weekly" in verdict.reason


@pytest.mark.parametrize("status", ["signed_out", "unsupported", "unavailable", ""])
def test_no_reading_is_unknown_and_never_a_clearance(status) -> None:
    verdict = quota_verdict(_usage(_window(5.0), status=status))
    assert verdict.unknown and not verdict.blocked


def test_an_ok_seat_with_no_numbers_is_unknown_too() -> None:
    assert quota_verdict(_usage()).unknown


def test_a_nonsense_percentage_is_not_a_number() -> None:
    """A provider payload this build cannot read must not become a decision."""
    bad = UsageWindow(kind="session", percent=float("nan"), severity="normal")
    assert quota_verdict(_usage(bad)).unknown


def test_a_cached_reading_still_decides_and_says_so() -> None:
    """Stale is not useless — but the receipt names where the number came from."""
    verdict = quota_verdict(_usage(_window(96.0), source="cached"))
    assert verdict.blocked and "cached" in verdict.reason


# ------------------------------------------------------------ over many seats


def test_one_exhausted_seat_blocks_even_beside_a_fresh_one() -> None:
    verdict = may_delegate([_usage(_window(5.0)), _usage(_window(99.0))])
    assert verdict.blocked


def test_one_seat_with_room_is_enough_when_nothing_is_exhausted() -> None:
    verdict = may_delegate([_usage(status="unsupported"), _usage(_window(20.0))])
    assert verdict.state == "allowed"


def test_no_seats_at_all_is_unknown() -> None:
    assert may_delegate([]).unknown
    assert may_delegate(None).unknown


# ------------------------------------------------------- the turn's own stop


async def test_a_request_that_keeps_delegating_is_stopped_by_its_budget(
    tmp_path, monkeypatch
):
    """Three coding agents per request is the ceiling; the fourth wave stops.

    ``MAX_WAVES`` cannot catch this: each wave here delivers a result and
    would legitimately earn another. What must stop the turn is the number of
    seats it has spent, with a named reason rather than a quiet end.
    """
    from jarvis.agent_chat import delegation_wait
    from jarvis.agent_chat import service as service_mod
    from jarvis.agent_chat.events import make_event
    from jarvis.agent_chat.service import AgentChatService
    from jarvis.agent_chat.store import AgentChatStore
    from jarvis.core.delegated_work import DelegatedWork, register_delegated_work

    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    store = AgentChatStore(tmp_path / "chat.db")
    svc = AgentChatService(store)
    session = store.create_session(
        provider="openai", model="", effort="", surface="jarvis", cwd=str(tmp_path)
    )
    waves = 0

    async def runner(handle, prompt, *args, **kwargs):
        nonlocal waves
        waves += 1
        # Every wave hands work to one more coding pane, and it finishes at once.
        index = waves

        async def probe():
            return {"status": "completed", "report": f"pane {index} done"}

        register_delegated_work(DelegatedWork(f"coding:t{index}:1:{index}", f"T{index}", probe))
        await handle.emit(
            make_event("turn_finished", {"turn_id": handle.turn_id, "status": "done", "usage": {}})
        )

    monkeypatch.setattr(service_mod, "run_brain_turn", runner)
    monkeypatch.setattr(service_mod, "resolve_runner", lambda *a, **kw: "brain")
    await svc.send(session.session_id, "Keep going on Bubble until it is finished.")
    await asyncio.wait_for(asyncio.shield(svc._running[session.session_id].task), 10)

    events = store.list_events(session.session_id)
    finished = [e["payload"] for e in events if e["kind"] == "turn_finished"]
    assert finished[-1]["error"] == "delegation_budget_exhausted", finished[-1]
    # Three delegations ran; the wave that would have opened a fourth did not.
    assert waves == MAX_CODING_DELEGATIONS, waves
