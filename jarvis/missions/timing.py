"""Phase timings for the stretch between a step starting and its worker running.

Live 2026-10-25, mission ``01a17410``: 71.9 s passed between
``WorktreeManager.create`` and the worker being chosen, and the log held
exactly one unrelated line in that whole window. Measured offline afterwards,
the obvious suspects account for almost none of it — the base-SHA write,
the worker contract, the learned-rules lookup and the prompt assembly together
come to about 74 ms — so roughly 50 s has no owner at all.

This records how long each phase took. Nothing else: no payload, no prompt,
no path, no identity, only a phase name, a short reference and a duration.

Two rules follow from what the numbers are for. A phase is attributed by
subtracting its parts from its whole, so **an absent line is not a zero**: a
phase under the noise floor still reports, at DEBUG, and only the INFO stream
is abridged. And a wait that a reader would otherwise have to derive is
measured directly instead — see :func:`timed_offload`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

#: Below this a phase is noise against the ~50 s being hunted, and logging it
#: at INFO every iteration would bury the phases that matter. It drops to
#: DEBUG rather than disappearing — a reader subtracting parts from a whole
#: needs every part, and a gap it has to guess at is worse than a small number.
_FLOOR_MS = 50.0

_T = TypeVar("_T")


def _emit(
    phase: str,
    ref: str,
    elapsed_ms: float,
    *,
    failed: bool = False,
    always: bool = False,
) -> None:
    """Report one measured phase, at INFO when it is worth reading."""
    level = (
        logging.INFO
        if failed or always or elapsed_ms >= _FLOOR_MS
        else logging.DEBUG
    )
    logger.log(
        level,
        "mission timing: %s %s %.0f ms%s",
        ref, phase, elapsed_ms, " (failed)" if failed else "",
    )


@contextmanager
def timed(phase: str, ref: str, *, always: bool = False) -> Iterator[None]:
    """Log how long ``phase`` took, for the step ``ref`` identifies.

    ``always`` keeps a fast phase at INFO, for one that is expected to be quick
    and whose silence in that stream would be ambiguous. A raising phase is
    timed too and says so — a failure that takes 40 s is exactly what this is
    for.
    """
    started = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _emit(phase, ref, elapsed_ms, failed=failed, always=always)


async def timed_offload(
    phase: str,
    ref: str,
    fn: Callable[..., _T],
    /,
    **kwargs: Any,
) -> _T:
    """``asyncio.to_thread(fn, **kwargs)`` with the queue wait measured, not derived.

    Offloading hides two different delays behind one number: the time the call
    spent queued for a free thread in the default executor, and the time the
    work itself took. Subtracting the inner phases from the outer one to
    recover the first is exactly the arithmetic that an abridged log breaks, so
    the handoff is timed where it happens — between submitting the call and the
    worker thread entering it. Both numbers always reach INFO.

    A call that never reached a thread (the await was cancelled while queued)
    reports its wait as failed, because that duration is a lower bound only.
    """
    submitted = time.perf_counter()
    entered: float | None = None

    def _run() -> _T:
        nonlocal entered
        entered = time.perf_counter()
        return fn(**kwargs)

    try:
        return await asyncio.to_thread(_run)
    finally:
        done = time.perf_counter()
        _emit(
            f"{phase}.queue_wait",
            ref,
            ((entered if entered is not None else done) - submitted) * 1000.0,
            failed=entered is None,
            always=True,
        )
        _emit(phase, ref, (done - submitted) * 1000.0, always=True)


__all__ = ["timed", "timed_offload"]
