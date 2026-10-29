"""What one request may spend on coding agents, and when to stop asking.

Two limits, deliberately separate, because they fail for different reasons.

**How many coding agents one request may start.** ``MAX_WAVES`` already bounds
how often a turn may come back for more results; it does not bound how much
*work* one request hands out, and those are not the same number. A request
that keeps finding "one more small thing" can open a new pane on every wave,
and each pane is a real seat spending a real subscription window. So a single
original request gets a small, conservative ceiling on coding delegations —
reached, the turn stops with a named reason instead of starting another agent.

**Whether there is any budget left at all.** :mod:`jarvis.agent_usage` already
answers this honestly and cheaply for each connected seat: one cached HTTPS
read, no pane spawned, no model called, and every reading carries whether it
is live or cached and what it was true at. This module turns that into one
decision — may a new delegation start? — with three outcomes and no fourth:

``allowed``
    A trustworthy reading leaves room.
``blocked``
    A trustworthy reading says ten percent or less is left. Nothing new is
    started; running work is not touched.
``unknown``
    No usable reading. The caller proceeds and SAYS the quota is unknown. A
    percentage is never guessed, and a dollar figure is never converted into
    one — a cost is not a window.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Coding delegations one original user request may start. Small on purpose:
#: a chief that needs a fourth agent for one request has misunderstood it.
MAX_CODING_DELEGATIONS = 3

#: Remaining share of a plan window below which no NEW delegation starts.
STOP_BELOW_REMAINING_PERCENT = 10.0

#: How long a usage reading may be before it stops being a clearance.
MAX_READING_AGE_S = 15 * 60.0

#: The refusal a dispatch gets once its request has spent its budget.
BUDGET_EXHAUSTED = "delegation_budget_exhausted"

#: The work-key prefix ``jarvis/agentic_ide/delegation_wait.py`` registers.
CODING_PREFIX = "coding:"


class DelegationLedger:
    """Slots one original request may spend on coding agents, reserved up front.

    Counting handed-back work was the wrong moment twice over. It let a single
    wave start any number of panes before anything was counted, and when the
    count was finally reached it ended the whole turn -- including the
    independent verification the last result still needed. A limit has to bite
    where the work STARTS and nowhere else.

    Reservations are keyed by the request, not the turn: a result that comes
    back after its turn closed continues the same request
    (``ChatTurn.request_id``), and must not be handed a fresh budget.
    ``request_id`` makes a reservation idempotent, so a replayed or retried
    dispatch of the SAME assignment keeps its own slot instead of taking a
    second one.
    """

    def __init__(self, limit: int = MAX_CODING_DELEGATIONS) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._slots: dict[tuple[str, str], dict[str, float]] = {}

    def reserve(self, key: tuple[str, str], request_id: str, *, limit: int | None = None) -> bool:
        """Take a slot for ``request_id``; ``False`` means the budget is spent.

        Re-reserving the same ``request_id`` always succeeds: it already owns
        its slot, and a dispatch that is retried under its own id is one
        delegation, not two.
        """
        ceiling = self._limit if limit is None else limit
        with self._lock:
            taken = self._slots.setdefault(key, {})
            if request_id and request_id in taken:
                return True
            if len(taken) >= ceiling:
                return False
            taken[request_id or f"anon:{len(taken)}"] = time.time()
            return True

    def release(self, key: tuple[str, str], request_id: str) -> None:
        """Give a slot back when the dispatch it was taken for never started."""
        with self._lock:
            taken = self._slots.get(key)
            if taken is not None:
                taken.pop(request_id, None)
                if not taken:
                    self._slots.pop(key, None)

    def spent(self, key: tuple[str, str]) -> int:
        with self._lock:
            return len(self._slots.get(key, {}))

    def forget(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._slots.pop(key, None)


#: The process-wide ledger. One chat's budget is per request, so a single
#: instance is enough and keeps a replayed dispatch idempotent across turns.
LEDGER = DelegationLedger()


def request_key(turn: Any) -> tuple[str, str] | None:
    """The budget key of a :class:`jarvis.core.chat_turn.ChatTurn`, if it owns one."""
    session_id = str(getattr(turn, "session_id", "") or "")
    request = str(getattr(turn, "request_id", "") or getattr(turn, "turn_id", "") or "")
    return (session_id, request) if session_id and request else None


def coding_delegations(keys: Any) -> int:
    """How many distinct coding agents a set of work keys names."""
    return len({str(key) for key in keys or () if str(key).startswith(CODING_PREFIX)})


@dataclass(frozen=True)
class QuotaVerdict:
    """One decision about starting new paid work, with its own evidence."""

    state: str  # "allowed" | "blocked" | "unknown"
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.state == "blocked"

    @property
    def unknown(self) -> bool:
        return self.state == "unknown"

    @property
    def allowed(self) -> bool:
        return self.state == "allowed"


def _remaining(window: Any) -> float | None:
    """What a window leaves, from the USED percentage it reports."""
    percent = getattr(window, "percent", None)
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        return None
    number = float(percent)
    # NaN / inf reach here from a provider payload this build cannot read.
    # They must stay "no reading", never become a decision.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return 100.0 - number


def quota_verdict(
    usage: Any,
    *,
    floor: float = STOP_BELOW_REMAINING_PERCENT,
    max_age_s: float = MAX_READING_AGE_S,
    now: float | None = None,
) -> QuotaVerdict:
    """Whether ONE seat's own reading permits starting new work on that seat.

    ``usage`` is an :class:`jarvis.agent_usage.AccountUsage`. Only ``ok`` with
    at least one numeric window is a reading; ``signed_out``, ``unsupported``
    and ``unavailable`` are not bad news, they are no news, and neither a
    block nor a clearance.

    Age cuts both ways and not symmetrically. A stale reading may still
    BLOCK -- a window that was nearly empty a while ago has not refilled by
    being ignored -- but it may never CLEAR, because the only thing that
    happens to a budget while nobody looks is that it gets smaller.
    """
    status = str(getattr(usage, "status", "") or "")
    if status != "ok":
        return QuotaVerdict("unknown", f"the seat reports {status or 'nothing'}")
    windows = tuple(getattr(usage, "windows", ()) or ())
    left = [value for value in (_remaining(w) for w in windows) if value is not None]
    if not left:
        return QuotaVerdict("unknown", "the seat published no usable window")
    tightest = min(left)
    source = str(getattr(usage, "source", "") or "live")
    if tightest <= floor:
        kind = next(
            (
                str(getattr(w, "kind", "") or "window")
                for w in windows
                if _remaining(w) == tightest
            ),
            "window",
        )
        return QuotaVerdict(
            "blocked",
            f"{tightest:.0f}% of this seat's {kind} budget is left ({source} reading); "
            "no new coding agent was started",
        )
    as_of = getattr(usage, "as_of", None)
    if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
        age = (time.time() if now is None else now) - float(as_of)
        if age > max_age_s:
            return QuotaVerdict(
                "unknown",
                f"this seat's reading is {age / 60:.0f} minutes old; it says "
                f"{tightest:.0f}% was left then, which is not a clearance now",
            )
    elif source != "live":
        return QuotaVerdict("unknown", "a cached reading of unknown age is not a clearance")
    return QuotaVerdict("allowed", f"{tightest:.0f}% left")


def verdict_for_seat(
    readings: Any,
    account_id: str,
    *,
    floor: float = STOP_BELOW_REMAINING_PERCENT,
    now: float | None = None,
) -> QuotaVerdict:
    """The verdict for the seat the work would actually run on.

    Judging a *set* of seats was wrong in both directions: a nearly empty
    seat the dispatch does not use would have blocked a healthy one, and a
    healthy seat elsewhere would have cleared one whose own budget is
    unknown. Only the selected ``account_id`` decides, and a dispatch with no
    named seat is ``unknown`` rather than fine.
    """
    if not account_id:
        return QuotaVerdict("unknown", "the dispatch named no seat")
    rows = readings.values() if isinstance(readings, Mapping) else (readings or ())
    for reading in rows:
        if str(getattr(reading, "account_id", "") or "") == account_id:
            return quota_verdict(reading, floor=floor, now=now)
    return QuotaVerdict("unknown", f"no usage reading for seat {account_id}")


__all__ = [
    "BUDGET_EXHAUSTED",
    "CODING_PREFIX",
    "LEDGER",
    "MAX_CODING_DELEGATIONS",
    "MAX_READING_AGE_S",
    "STOP_BELOW_REMAINING_PERCENT",
    "DelegationLedger",
    "QuotaVerdict",
    "coding_delegations",
    "quota_verdict",
    "request_key",
    "verdict_for_seat",
]
