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
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Coding delegations one original user request may start. Small on purpose:
#: a chief that needs a fourth agent for one request has misunderstood it.
MAX_CODING_DELEGATIONS = 3

#: Remaining share of a plan window below which no NEW delegation starts.
STOP_BELOW_REMAINING_PERCENT = 10.0

#: The error a turn ends with when it has spent its delegation budget.
BUDGET_EXHAUSTED = "delegation_budget_exhausted"

#: The work-key prefix ``jarvis/agentic_ide/delegation_wait.py`` registers.
CODING_PREFIX = "coding:"


def coding_delegations(keys: Any) -> int:
    """How many distinct coding agents this request has handed work to."""
    return len({str(key) for key in keys or () if str(key).startswith(CODING_PREFIX)})


def budget_spent(keys: Any, *, limit: int = MAX_CODING_DELEGATIONS) -> bool:
    """Whether this request may not start another coding agent."""
    return coding_delegations(keys) >= limit


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
    usage: Any, *, floor: float = STOP_BELOW_REMAINING_PERCENT
) -> QuotaVerdict:
    """Whether one seat's own reading permits starting new work.

    ``usage`` is an :class:`jarvis.agent_usage.AccountUsage`. Only ``ok`` with
    at least one numeric window is a reading; ``signed_out``, ``unsupported``
    and ``unavailable`` are not bad news, they are no news, and they must not
    be reported as either a block or a clearance.
    """
    status = str(getattr(usage, "status", "") or "")
    if status != "ok":
        return QuotaVerdict("unknown", f"the seat reports {status or 'nothing'}")
    windows = tuple(getattr(usage, "windows", ()) or ())
    left = [value for value in (_remaining(w) for w in windows) if value is not None]
    if not left:
        return QuotaVerdict("unknown", "the seat published no usable window")
    tightest = min(left)
    if tightest <= floor:
        kind = next(
            (
                str(getattr(w, "kind", "") or "window")
                for w in windows
                if _remaining(w) == tightest
            ),
            "window",
        )
        source = str(getattr(usage, "source", "") or "live")
        return QuotaVerdict(
            "blocked",
            f"{tightest:.0f}% of this seat's {kind} budget is left ({source} reading); "
            "no new coding agent was started",
        )
    return QuotaVerdict("allowed", f"{tightest:.0f}% left")


def may_delegate(readings: Any, *, floor: float = STOP_BELOW_REMAINING_PERCENT) -> QuotaVerdict:
    """The verdict over every seat that could take the work.

    A block beats everything: if any trustworthy reading is that low, the seat
    it belongs to must not be handed more. Otherwise one clearance is enough,
    and with no reading at all the answer stays ``unknown`` — never "fine".
    """
    verdicts = [quota_verdict(reading, floor=floor) for reading in (readings or ())]
    for verdict in verdicts:
        if verdict.blocked:
            log.info("delegation budget: refusing new work — %s", verdict.reason)
            return verdict
    for verdict in verdicts:
        if verdict.state == "allowed":
            return verdict
    return QuotaVerdict("unknown", "no seat published a usable budget reading")


__all__ = [
    "BUDGET_EXHAUSTED",
    "CODING_PREFIX",
    "MAX_CODING_DELEGATIONS",
    "STOP_BELOW_REMAINING_PERCENT",
    "QuotaVerdict",
    "budget_spent",
    "coding_delegations",
    "may_delegate",
    "quota_verdict",
]
