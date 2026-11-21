"""A rejected handback becomes a documented failure, and one short review.

:mod:`jarvis.agent_chat.delegation_acceptance` judges and nothing more --
"it approves nothing, runs nothing, sends nothing and retries nothing". That is
the right contract for a verdict, but a rejection that is only spoken teaches
nobody: the next job repeats it. So the rejection is also written down, through
the same pipeline the missions already use
(:class:`jarvis.missions.conductor.incidents.IncidentRecorder`):

1. the incident goes to ``incidents.jsonl`` with no model call at all, so it
   survives every provider being down;
2. the FIRST rejection of each job -- never each attempt (AP-19) -- starts ONE
   completion on the conductor's subscription seat, which may write a rule;
3. the strongest rules for that lane are put in front of the next worker
   (``lessons.lessons_block``), so the same mistake is named before it is made.

What this does NOT do is grant anything either. The gate's verdict stands
exactly as it was, the answer the person reads is unchanged, and no work is
re-dispatched here. A review's suggestion is input for a later attempt, never
evidence that anything was fixed.

The recorder is absent before the mission runtime has booted and in tests that
do not install one; documenting then degrades to a no-op, because the chat turn
must never fail over its own bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Keeps one incident line readable; the full order lives in the chat history.
_REQUEST_CHARS = 500


def _executor_of(gate: Any) -> str:
    """Which workspace(s) produced the rejected handback, for the record."""
    seen: list[str] = []
    for job in getattr(gate, "jobs", ()) or ():
        name = str(getattr(job, "workspace_id", "") or "")
        if name and name not in seen:
            seen.append(name)
    return ",".join(seen)


def _cause_of(gate: Any, gaps: tuple[str, ...]) -> str:
    """One line naming the jobs, what they said about themselves, and what is missing."""
    states: list[str] = []
    for job in getattr(gate, "jobs", ()) or ():
        state = str(getattr(job, "status", "") or "unknown")
        if state not in states:
            states.append(state)
    said = "/".join(states) if states else "unknown"
    missing = ", ".join(gaps) if gaps else "no evidence recorded"
    work = list(getattr(gate, "work", ()))
    return f"acceptance gate rejected {work} (reported {said}): {missing}"


async def document_rejection(gate: Any, ledger: Any, request: str) -> bool:
    """Write the gate's rejection down and let the seat learn once per job.

    Returns whether an incident was recorded, which is what the tests assert;
    callers ignore it. Never raises: a chat turn is not allowed to fail because
    its own bookkeeping did.
    """
    try:
        from jarvis.missions.conductor.incidents import active_recorder
        from jarvis.missions.conductor.lanes import first_lane
        from jarvis.missions.conductor.lessons import Incident

        recorder = active_recorder()
        if recorder is None:
            # Before boot, or a surface that runs without the mission runtime.
            # Said out loud so a missing lesson is never a silent one (AP-30).
            log.info("delegation rejection not documented: no incident recorder active")
            return False

        work = tuple(getattr(gate, "work", ()) or ())
        gaps = tuple(gate.gaps(ledger))
        order = (request or "").strip()
        incident = Incident(
            lane=first_lane(order) if order else "unknown",
            executor=_executor_of(gate),
            cause=_cause_of(gate, gaps),
            request=order[:_REQUEST_CHARS],
        )
        # The job, not the attempt, is the unit of work: a second rejection of
        # the same job must not buy a second review.
        await recorder.record(incident, learn_key="|".join(sorted(work)))
        return True
    except Exception:  # noqa: BLE001 - the verdict already stands; this is bookkeeping
        log.warning("Could not document the delegation rejection", exc_info=True)
        return False


__all__ = ["document_rejection"]
