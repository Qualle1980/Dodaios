"""What one request may spend on coding agents, and when the answer is "unknown".

The review of 2026-10-29 found the first cut counted the wrong thing at the
wrong time: it counted work that had already come BACK, which let one wave
start any number of panes and then ended the whole turn -- including the
verification the last result still needed. A limit has to bite where work
starts.
"""

from __future__ import annotations

import time

import pytest

from jarvis.agent_chat.delegation_budget import (
    MAX_CODING_DELEGATIONS,
    DelegationLedger,
    coding_delegations,
    quota_verdict,
    request_key,
    verdict_for_seat,
)
from jarvis.agent_usage import AccountUsage, UsageWindow
from jarvis.core.chat_turn import ChatTurn

KEY = ("chat-1", "turn-1")


def _window(used: float, *, kind: str = "session") -> UsageWindow:
    return UsageWindow(kind=kind, percent=used, severity="normal")


def _usage(
    *windows: UsageWindow,
    status: str = "ok",
    source: str = "live",
    account_id: str = "seat-a",
    as_of: float | None = None,
) -> AccountUsage:
    return AccountUsage(
        account_id=account_id,
        platform="claude",
        status=status,
        windows=windows,
        source=source,
        as_of=time.time() if as_of is None else as_of,
    )


# -------------------------------------------------------- reserving up front


def test_a_request_gets_exactly_its_slots() -> None:
    ledger = DelegationLedger(limit=3)
    assert [ledger.reserve(KEY, f"r{n}") for n in range(4)] == [True, True, True, False]
    assert ledger.spent(KEY) == 3


def test_four_parallel_dispatches_admit_only_three() -> None:
    """Four dispatches racing for the same request: the fourth is refused.

    The reservation is taken under a lock before anything starts, so the
    outcome does not depend on which dispatch gets there first.
    """
    ledger = DelegationLedger(limit=3)
    granted = [ledger.reserve(KEY, f"req-{n}") for n in range(4)]
    assert granted.count(True) == 3 and granted.count(False) == 1


def test_replaying_the_same_request_id_does_not_consume_a_second_slot() -> None:
    """Idempotent by ``request_id``: a retried dispatch is one delegation."""
    ledger = DelegationLedger(limit=3)
    for _ in range(5):
        assert ledger.reserve(KEY, "same-request")
    assert ledger.spent(KEY) == 1


def test_a_dispatch_that_never_started_gives_its_slot_back() -> None:
    ledger = DelegationLedger(limit=1)
    assert ledger.reserve(KEY, "r1")
    assert not ledger.reserve(KEY, "r2")
    ledger.release(KEY, "r1")
    assert ledger.reserve(KEY, "r2")


def test_another_request_has_its_own_budget() -> None:
    ledger = DelegationLedger(limit=1)
    assert ledger.reserve(KEY, "r1")
    assert ledger.reserve(("chat-1", "turn-2"), "r1")
    assert ledger.reserve(("chat-2", "turn-1"), "r1")


def test_the_budget_follows_the_request_not_the_turn() -> None:
    """A result answered in a LATER turn must not be handed a fresh budget.

    ``parked_work`` answers a delegated result in a turn of its own; keying on
    ``turn_id`` would have reset the limit every time that happened.
    """
    first = ChatTurn("chat-1", "turn-1", "Work on Bubble.", True, "tr")
    later = ChatTurn("chat-1", "turn-9", "continue", True, "tr", origin_turn_id="turn-1")
    assert request_key(first) == request_key(later) == ("chat-1", "turn-1")

    ledger = DelegationLedger(limit=1)
    assert ledger.reserve(request_key(first), "r1")
    assert not ledger.reserve(request_key(later), "r2")


def test_a_turn_without_an_identity_owns_no_budget() -> None:
    assert request_key(None) is None
    assert request_key(ChatTurn("", "", "", True, "")) is None


def test_only_coding_work_counts_as_a_delegation() -> None:
    keys = ["coding:t1:1:2", "dispatch:4711", "coding:t2:1:3", "mission:x", "coding:t1:1:2"]
    assert coding_delegations(keys) == 2
    assert coding_delegations(None) == 0


def test_the_shipped_ceiling_is_small_on_purpose() -> None:
    assert MAX_CODING_DELEGATIONS == 3


# ------------------------------------------------------------ quota readings


def test_a_full_window_allows_new_work() -> None:
    assert quota_verdict(_usage(_window(12.0))).allowed


@pytest.mark.parametrize("used", [90.0, 95.0, 100.0])
def test_ten_percent_or_less_left_blocks_new_work(used) -> None:
    verdict = quota_verdict(_usage(_window(used)))
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


def test_an_ok_seat_with_no_numbers_is_unknown() -> None:
    assert quota_verdict(_usage()).unknown


def test_a_nonsense_percentage_is_not_a_number() -> None:
    """A payload this build cannot read must not become a decision."""
    bad = UsageWindow(kind="session", percent=float("nan"), severity="normal")
    assert quota_verdict(_usage(bad)).unknown


def test_an_old_reading_may_still_block_but_may_not_clear() -> None:
    """The only thing that happens to a budget while nobody looks is shrinking."""
    long_ago = time.time() - 3600
    assert quota_verdict(_usage(_window(96.0), as_of=long_ago)).blocked
    stale = quota_verdict(_usage(_window(20.0), as_of=long_ago))
    assert stale.unknown and "old" in stale.reason


def test_a_cached_reading_without_a_timestamp_is_not_a_clearance() -> None:
    usage = AccountUsage(
        account_id="seat-a", platform="claude", status="ok",
        windows=(_window(20.0),), source="cached", as_of=None,
    )
    assert quota_verdict(usage).unknown


# ------------------------------------------------------- the seat that is used


def test_only_the_selected_seat_decides() -> None:
    readings = {
        # 95 % used leaves 5 %; 20 % used leaves plenty.
        "seat-a": _usage(_window(95.0), account_id="seat-a"),
        "seat-b": _usage(_window(20.0), account_id="seat-b"),
    }
    assert verdict_for_seat(readings, "seat-a").blocked
    assert verdict_for_seat(readings, "seat-b").allowed


def test_a_healthy_target_is_not_blocked_by_an_unselected_seat() -> None:
    """Judging a set of seats blocked a good target because another was low."""
    readings = [_usage(_window(99.0), account_id="other"), _usage(_window(10.1), account_id="mine")]
    assert verdict_for_seat(readings, "mine").allowed


def test_an_unknown_target_is_not_cleared_by_another_seat() -> None:
    readings = [
        _usage(_window(10.0), account_id="other"),
        _usage(status="unsupported", account_id="mine"),
    ]
    assert verdict_for_seat(readings, "mine").unknown


def test_a_seat_with_no_reading_at_all_is_unknown() -> None:
    assert verdict_for_seat([], "mine").unknown
    assert verdict_for_seat(None, "mine").unknown


def test_a_dispatch_that_names_no_seat_is_unknown_not_fine() -> None:
    assert verdict_for_seat([_usage(_window(99.0))], "").unknown
