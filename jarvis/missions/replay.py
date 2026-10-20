"""Rebuild a resumable checkpoint from a mission's own event log.

A capacity park writes ``checkpoint.json`` and the resume path reads it. A
hard crash writes nothing, so the orchestrator used to lose every approved
step of an interrupted mission: recovery swept it ``crash_recovery`` and a
re-run re-dispatched the prompt as a NEW mission from zero — the same work
again, on the same subscription, for a second time.

Nothing needed to be written more often to fix that. The mission event log is
already the authoritative, append-only record of the run: ``MissionPlanReady``
carries the full plan, ``WorkerSpawned`` says which step a worker took, and an
approving ``CriticVerdictReady`` says that step passed review. That is exactly
the information a checkpoint holds, so it can be replayed out of the log
instead of being duplicated onto the hot path — the same approach durable
execution engines take, where the event history IS the recovery mechanism
rather than a snapshot beside it.

What a replayed checkpoint deliberately does NOT claim:

* **no provider pin** — the family a crashed run was waiting for is not a fact
  the log records, and guessing one would pin a resume to the wrong plan;
* **no backoff** — a crash is not a spent window, so the resume may start at
  once;
* **nothing pending review** — an unreviewed draft is re-reviewed rather than
  assumed approved;
* **paid spend** is not copied: the ledger is read from ``MissionPaidUsage``
  events by the caller, which is the same source either way.

Pure functions over event payloads. No I/O, no clock, no provider call.
"""
from __future__ import annotations

from typing import Any

#: Mission states a crashed run can be parked FROM (state_machine.py allows
#: exactly these three into WAITING_CAPACITY). PENDING never has an approved
#: step, so it is not among them.
RESUMABLE_FROM: frozenset[str] = frozenset({"RUNNING", "CRITIQUING", "LOOPING"})

#: Reason stamped on the state change when a crashed mission is re-parked, and
#: counted in the log to bound how often that may happen.
CRASH_RESUME_REASON = "crash_resumable"

#: How often one mission may be re-parked after a crash before it is failed
#: honestly. A mission that takes the app down with it must not come back on
#: every boot — that is an automatic retry loop, which this project forbids.
MAX_CRASH_RESUMES = 2


def _payload(event: Any) -> Any:
    return getattr(event, "payload", None)


def _event_type(payload: Any) -> str:
    return str(getattr(payload, "event_type", "") or "")


def plan_from_events(events: list[Any]) -> dict[str, Any] | None:
    """The plan dict (``MissionPlan`` shape) the mission was running, or None.

    The LAST ``MissionPlanReady`` wins: a re-planned mission runs its newest
    plan, and an approved step from an older plan simply no longer matches a
    task id in it.
    """
    plan: dict[str, Any] | None = None
    for event in events:
        payload = _payload(event)
        if _event_type(payload) != "MissionPlanReady":
            continue
        steps = getattr(payload, "plan", None)
        if not isinstance(steps, list) or not steps:
            continue
        plan = {
            "steps": [dict(step) for step in steps if isinstance(step, dict)],
            "n_workers": max(1, int(getattr(payload, "n_workers", 0) or 1)),
            "expected_output": str(getattr(payload, "expected_output", "") or ""),
        }
    if plan is not None and not plan["steps"]:
        return None
    return plan


def approved_task_ids(events: list[Any]) -> set[str]:
    """Task ids whose review passed, read from the log.

    A worker is tied to its step by ``WorkerSpawned.step``, and the verdict is
    tied to the worker by ``CriticVerdictReady.worker_id``. A later revise on
    the same step does not un-approve it: the critic loop only grades again
    when the step was NOT approved.
    """
    worker_to_task: dict[str, str] = {}
    approved: set[str] = set()
    for event in events:
        payload = _payload(event)
        kind = _event_type(payload)
        if kind == "WorkerSpawned":
            step = getattr(payload, "step", None)
            task_id = str((step or {}).get("task_id") or "") if isinstance(step, dict) else ""
            worker_id = str(getattr(payload, "worker_id", "") or "")
            if task_id and worker_id:
                worker_to_task[worker_id] = task_id
        elif kind == "CriticVerdictReady":
            if str(getattr(payload, "verdict", "")) != "approve":
                continue
            worker_id = str(getattr(payload, "worker_id", "") or "")
            task_id = worker_to_task.get(worker_id)
            if task_id:
                approved.add(task_id)
    return approved


def crash_resume_count(events: list[Any]) -> int:
    """How often this mission has already been re-parked after a crash."""
    return sum(
        1
        for event in events
        if _event_type(_payload(event)) == "MissionStateChanged"
        and str(getattr(_payload(event), "reason", "")) == CRASH_RESUME_REASON
    )


def replay_checkpoint(
    mission_id: str, prompt: str, events: list[Any]
) -> dict[str, Any] | None:
    """A checkpoint equivalent for a crashed mission, or None when there is none.

    None means "nothing worth resuming": no plan in the log, no step approved,
    every step already approved (the mission was finishing, so the normal
    reconcile/sweep is the honest answer), or the crash-resume bound is spent.
    """
    plan = plan_from_events(events)
    if plan is None:
        return None
    task_ids = [
        str(step.get("task_id") or "") for step in plan["steps"] if step.get("task_id")
    ]
    done = approved_task_ids(events) & set(task_ids)
    if not done:
        return None  # a re-run loses nothing but time
    if len(done) == len(task_ids):
        return None  # all steps passed; this is a finalization, not a resume
    if crash_resume_count(events) >= MAX_CRASH_RESUMES:
        return None
    return {
        "mission_id": mission_id,
        "prompt": prompt,
        # Honest label from the closed capacity vocabulary: the process that
        # held this run is gone, so its executor is not available to continue.
        "reason": "provider_unavailable",
        "provider": None,
        "plan": plan,
        "steps": [{"task_id": task_id, "done": task_id in done} for task_id in task_ids],
        "replayed_from_events": True,
    }


__all__ = [
    "CRASH_RESUME_REASON",
    "MAX_CRASH_RESUMES",
    "RESUMABLE_FROM",
    "approved_task_ids",
    "crash_resume_count",
    "plan_from_events",
    "replay_checkpoint",
]
