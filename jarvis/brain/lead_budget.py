"""A tool lead gets a short deadline for its FIRST sign of life.

When the active talker cannot call tools, a tool-capable provider leads the
turn and picks the tool (``BrainManager._first_tool_capable_provider``). For a
subscription talker with no key slot, the seat policy narrows the candidates to
itself and keyless local runtimes — deliberately, because nothing the user did
not start may bill a key. That is the right money rule and it can hand the lead
to a local runtime that is reachable but not *ready*.

Measured live on 2026-11-06, a voice turn on a tool-less talker:

    20:29:50.1  Intelligent router: chatgpt-web cannot call tools -
                ollama leads this turn and picks the tool
    20:30:11.5  first call to the local runtime - 21 s later
    20:30:16.6  ollama: no model configured - using smallest capable download
    20:30:17.9  ollama profile: created bonsai-2-4b-jarvis-... from bonsai-2:4b
    20:30:19.9  Brain-Stream stalled (no progress for 30.0s) - speaking fallback

The user heard the timeout filler phrase. No brain ever produced an answer, so
the turn record kept an empty answer and no provider at all. Choosing the model
and building its profile happened INSIDE the provider's first call, so neither
of the bounds that exist could fire: ``ToolUseLoop.deadline_s`` is checked
between tool rounds and the first round never finished, and the stream stall
watchdog is 30 s because that is the right number for an answer — not for a
lead whose whole job is to name a tool.

So the bound belongs here, on the lead and nowhere else: a lead that has shown
no sign of life within ``LEAD_FIRST_SIGN_BUDGET_S`` is abandoned and the chain
goes on to the talker, which answers tool-less. This is the whole-chain budget
the serving literature keeps arriving at - a per-call limit each attempt
respects still lets the chain overrun what the person is waiting for.

Nothing here judges the provider. An abandoned lead is OUR deadline talking, so
the caller records no health failure, no cooldown and no dead-list entry: its
status dot stays what it was.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

logger = logging.getLogger(__name__)

#: How long a lead may stay silent before the turn goes on without it.
#:
#: Measured 2026-11-06 for the providers that can lead here: the probe reached
#: ``gemini-3.5-flash-lite`` in 0.6 s and ``chatgpt-plan`` in 2.1 s, and a
#: healthy lead's first tool proposal landed 2.7 s into its call. A local
#: runtime that still has to choose and build a model profile took 26 s. Six
#: seconds sits well clear of every healthy case and well short of the 30 s
#: stall window that used to be the only thing catching this.
LEAD_FIRST_SIGN_BUDGET_S: Final[float] = 6.0


class LeadTookTooLong(Exception):
    """The lead showed no sign of life inside its budget; the chain moves on.

    Deliberately NOT a provider failure: it carries our own deadline, not a
    verdict about the provider, and the caller must not record it as one.
    """

    def __init__(self, provider: str, model: str | None, budget_s: float) -> None:
        super().__init__(
            f"{provider} led this turn and produced nothing within "
            f"{budget_s:.1f}s; continuing without a tool lead."
        )
        self.provider = provider
        self.model = model
        self.budget_s = budget_s


class LeadActivity:
    """One turn's "the lead is alive" flag.

    Any sign counts — a progress note, a tool acknowledgement, a text chunk.
    The point is not what the lead produced but that it is running rather than
    setting itself up.
    """

    __slots__ = ("_at", "_seen")

    def __init__(self) -> None:
        self._seen = False
        self._at: float | None = None

    def mark(self) -> None:
        """Record a sign of life. Idempotent and safe from any callback."""
        if not self._seen:
            self._seen = True
            self._at = time.monotonic()

    @property
    def seen(self) -> bool:
        return self._seen

    def wrap(self, callback: Any) -> Any:
        """``callback`` with a ``mark()`` in front of it, or a bare marker.

        The manager hands its progress and acknowledgement hooks through here,
        so a lead that is working is detected without the chain loop having to
        know what those hooks mean. A ``None`` callback still yields a marker:
        the signal matters even where nothing was listening.
        """
        def _marked(*args: Any, **kwargs: Any) -> Any:
            self.mark()
            if callback is None:
                return None
            return callback(*args, **kwargs)

        return _marked


async def run_lead_under_budget(
    attempt: Callable[[], Awaitable[Any]],
    activity: LeadActivity,
    *,
    provider: str,
    model: str | None,
    budget_s: float = LEAD_FIRST_SIGN_BUDGET_S,
    poll_s: float = 0.05,
) -> Any:
    """Await ``attempt``, abandoning it if it stays silent past ``budget_s``.

    Once the lead has shown any sign of life the budget is gone for good: it
    bounds the *setup*, never the work. A lead that picked a slow tool keeps
    all the time the tool needs, which is the whole reason this is a
    first-sign budget and not a total one.

    Raises :class:`LeadTookTooLong` after cancelling the attempt. Cancellation
    is awaited, so the provider's own cleanup runs before the chain continues.
    """
    task = asyncio.ensure_future(attempt())
    started = time.monotonic()
    # ``asyncio.wait`` reports back instead of raising on its timeout, so the
    # whole watch needs no exception handler: a poll that simply expired is the
    # normal path here, not a failure anyone should have to justify.
    while not activity.seen:
        remaining = budget_s - (time.monotonic() - started)
        if remaining <= 0.0:
            break
        done, _pending = await asyncio.wait({task}, timeout=min(poll_s, remaining))
        if done:
            # Finished inside the budget. ``await`` re-raises its own error, so
            # a real provider failure still reaches the chain as itself.
            return await task
    if activity.seen:
        return await task
    task.cancel()
    # ``asyncio.wait`` settles the task without re-raising anything, so the
    # abandonment needs no exception handler of its own: a cancelled attempt is
    # expected, and a failure that arrived after our deadline is not the cause
    # of anything the caller does next. It is still logged rather than lost.
    await asyncio.wait({task})
    if not task.cancelled():
        failure = task.exception()
        if failure is not None:
            logger.debug(
                "lead %s(%s) failed while being abandoned on our own budget: %s",
                provider, model, failure,
            )
    raise LeadTookTooLong(provider, model, budget_s)


__all__ = [
    "LEAD_FIRST_SIGN_BUDGET_S",
    "LeadActivity",
    "LeadTookTooLong",
    "run_lead_under_budget",
]
