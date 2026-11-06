"""The return path from a rejected handback: one more attempt, bounded.

:mod:`jarvis.agent_chat.delegation_acceptance` rejects an answer that calls
delegated coding work good without its own change and test evidence, and
:mod:`jarvis.agent_chat.delegation_lessons` writes that rejection down. Both
stop there on purpose. What was missing is the obvious next move: ask for the
missing evidence instead of ending the request.

**The boundary this module exists for.** ``delegation_acceptance.gate_for``
arms a job for EVERY handback, deliberately -- "a crashed, blocked or
superseded agent is further from acceptance, not closer to it". So a rejection
alone says nothing about whether anything may be re-dispatched. Only a job that
reports ``completed`` claims the work was handed back at all; a job waiting on a
permission decision (``needs_input``), a crashed one (``error`` /``cancelled``),
a replaced one (``superseded``) or one still working (``running``) is told to
the person honestly and never re-ordered on suspicion. That is an existing,
tested contract, not a preference.

**Why the bound is small and explicit.** Every correction wave is a real turn
and may drive a real coding agent, which costs the owner money per round. Two
attempts, counted per turn, and the counter lives next to the wave budget
rather than inside a prompt -- a model cannot be asked to limit itself
(AP-19 counts per unit of work).

**What a correction is not.** It carries the gaps and whatever rule the
conductor's seat learned from the first failure, so the next attempt knows what
was wrong before it starts. It grants nothing: the gate judges the corrected
answer exactly as it judged the first, and an attempt that still cannot show
evidence ends in the same honest answer as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jarvis.agent_chat.delegation_wait import _CONTINUATION_LEAD, MAX_CORRECTIONS

#: The one status that claims the work was handed back. Everything else is
#: further from acceptance and is never re-dispatched (see the module docstring).
DELIVERED = "completed"


def correctable(gate: Any) -> tuple[str, ...]:
    """The task ids of the jobs that claimed delivery, in a stable order."""
    jobs = getattr(gate, "jobs", ()) or () if gate is not None else ()
    return tuple(
        str(getattr(job, "task_id", "") or "")
        for job in jobs
        if str(getattr(job, "status", "") or "") == DELIVERED
        and str(getattr(job, "task_id", "") or "")
    )


def correction_prompt(
    *,
    gaps: Sequence[str],
    work: Sequence[str],
    original: str,
    rules: str,
    attempt: int,
) -> str:
    """The next attempt's briefing: what is missing, and what was learned.

    Opens with ``_CONTINUATION_LEAD`` so ``is_continuation_prompt`` keeps
    recognising it as composed text. A prompt that quotes the order must never
    be read as the person's own words, or it routes as a fresh instruction to
    the pane it names.
    """
    missing = "\n".join(f"- {gap}" for gap in gaps) or "- no evidence was recorded at all"
    learned = f"\nWhat earlier failures taught about this kind of work:\n{rules}\n" if rules else ""
    return (
        _CONTINUATION_LEAD
        + "The previous answer claimed this delegated coding work was good, and it was "
        "rejected: the evidence for it is missing. This is attempt "
        f"{attempt} of {MAX_CORRECTIONS} to close that gap.\n"
        f"Jobs: {list(work)}\nMissing evidence:\n{missing}\n"
        + learned
        + "Get the missing evidence yourself: read the change in that job's own "
        "checkout and run its tests there, or send the agent back for exactly what "
        "is missing. Do not claim the work is good until you have seen that "
        "evidence -- if you cannot get it, say plainly what is still unproven. "
        "Do not repeat assignments that are already finished.\n"
        "Original user request:\n" + original
    )


__all__ = ["DELIVERED", "MAX_CORRECTIONS", "correctable", "correction_prompt"]
