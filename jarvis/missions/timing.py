"""Phase timings for the stretch between a step starting and its worker running.

Live 2026-10-25, mission ``01a17410``: 71.9 s passed between
``WorktreeManager.create`` and the worker being chosen, and the log held
exactly one unrelated line in that whole window. Measured offline afterwards,
the obvious suspects account for almost none of it — the base-SHA write,
the worker contract, the learned-rules lookup and the prompt assembly together
come to about 74 ms — so roughly 50 s has no owner at all.

This records how long each phase took. Nothing else: no payload, no prompt,
no path, no identity, only a phase name, a short reference and a duration.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

#: Below this a phase is noise against the ~50 s being hunted, and logging it
#: every iteration would bury the phases that matter.
_FLOOR_MS = 50.0


@contextmanager
def timed(phase: str, ref: str, *, always: bool = False) -> Iterator[None]:
    """Log how long ``phase`` took, for the step ``ref`` identifies.

    ``always`` logs even a fast phase, for one that is expected to be quick
    and whose silence would otherwise be ambiguous. A raising phase is timed
    too and says so — a failure that takes 40 s is exactly what this is for.
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
        if failed or always or elapsed_ms >= _FLOOR_MS:
            logger.info(
                "mission timing: %s %s %.0f ms%s",
                ref, phase, elapsed_ms, " (failed)" if failed else "",
            )


__all__ = ["timed"]
