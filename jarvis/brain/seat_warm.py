"""Open the plan seat's conversation once after boot, so the first real
question does not pay for it.

Measured on 2026-10-20 against the running app: the FIRST question after a
start took ~12 s of browser setup on top of the answer — the web seat had to
open its effort menu (5.1 s), enter the prompt (2.1 s) and send (5.0 s) on a
fresh conversation. Every later question in the same session paid 1.5 s for
the same three steps, because the conversation was reused.

This sends ONE minimal turn after boot to create that conversation. It is OFF
by default and switched on per install (``[brain.routing] warm_seat_on_boot``):
a turn nobody asked for is a provider call nobody asked for, and that is only
acceptable where the person decided it is.

Narrow by construction:

* **plan seats only** — a key-billed provider is never warmed, so this can
  never spend money (``seat_bound``);
* **once per process**, with no retry: a failure means the first question pays
  what it pays today, which is the status quo, not an outage;
* **off the boot critical path** (AP-26) — the caller schedules it as a task
  and boot never waits for it.
"""
from __future__ import annotations

import logging
from typing import Any

from jarvis.core.protocols import BrainMessage, BrainRequest

log = logging.getLogger(__name__)

#: The shortest turn that still creates a conversation. It is never shown to
#: anyone: the reply is read and dropped.
WARM_PROMPT = "ok"

_warmed: set[str] = set()


def warm_seat_enabled(config: Any) -> bool:
    """The ONE reader of ``[brain.routing] warm_seat_on_boot`` (AP-31)."""
    routing = getattr(getattr(config, "brain", None), "routing", None)
    return bool(getattr(routing, "warm_seat_on_boot", False))


async def warm_chat_seat(config: Any, registry: Any) -> str:
    """Open the seat's conversation. Returns what happened, for the log."""
    from jarvis.brain.background_policy import seat_bound

    provider = str(getattr(getattr(config, "brain", None), "primary", "") or "")
    if not warm_seat_enabled(config):
        return "off"
    if not provider or not seat_bound(provider):
        # A key-billed provider pays per token: warming one would spend the
        # user's money on a turn they did not ask for.
        return f"skipped: {provider or 'no provider'} is not a plan seat"
    if provider in _warmed:
        return "already warmed this process"
    _warmed.add(provider)

    brain = registry.instantiate(provider)
    request = BrainRequest(
        messages=(BrainMessage(role="user", content=WARM_PROMPT),),
        system=None,
        stream=True,
        max_tokens=16,
    )
    async for _delta in brain.complete(request):
        pass
    return f"{provider} conversation opened"


def reset_for_tests() -> None:
    """Forget which providers were warmed (test seam only)."""
    _warmed.clear()


__all__ = ["WARM_PROMPT", "reset_for_tests", "warm_chat_seat", "warm_seat_enabled"]
