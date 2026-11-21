"""Write every failure down as it happens, without a model call.

:class:`IncidentRecorder` is the pipeline: the incident goes to disk first, and
only then may the conductor's seat learn from it. Two callers feed it.

* **The mission bus** (:func:`bind_incident_recorder`), like the budget
  tracker: a critic verdict that is not an approval and a failed mission each
  become one :class:`.lessons.Incident`, keyed by mission.
* **A rejected chat delegation**, which is NOT a mission and publishes nothing
  on that bus. The acceptance gate
  (:mod:`jarvis.agent_chat.delegation_acceptance`) rejects a handback whose
  evidence is missing, and that rejection has to be documented too. It reaches
  this module through :func:`active_recorder` and passes its own ``learn_key``
  (the job), because ``Incident.mission_id`` is empty for it and no mission is
  invented to carry it.

The record itself never needs a model, so nothing is lost when every provider
is down. With a ``learn`` callable, the FIRST incident of each unit of work
(a mission, or a job) starts ONE rule update
(:meth:`.lessons.LessonStore.lesson_prompt` -> ``learn`` ->
:meth:`.lessons.LessonStore.apply`) in the background -- per unit, never per
attempt (AP-19), or a stubborn job pays for a review every round. ``learn`` is
the conductor's subscription seat; it returns None when no subscription can
answer, and the incident then simply waits on disk (background policy: never a
key).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..events import CriticVerdictReady, EventEnvelope, MissionFailed
from ..stream_evidence import clean_request_body
from .lanes import first_lane
from .lessons import Incident, LessonStore

logger = logging.getLogger(__name__)

PromptLookup = Callable[[str], Awaitable[str | None]]
#: One completion on the conductor seat: prompt in, reply out, None = no seat.
LessonAsk = Callable[[str], Awaitable[str | None]]

#: Units remembered as "lesson already asked", so a revise loop costs one call.
_LEARNED_MEMORY = 256


class IncidentRecorder:
    """Document a failure, and let the seat learn from the first one per unit.

    Deliberately not bound to any bus: the chat delegation path has no mission
    event to publish. Construct one per store; ``learn`` may be None (headless,
    or no subscription configured), which leaves the incident on disk.
    """

    def __init__(self, lessons: LessonStore, learn: LessonAsk | None = None) -> None:
        self._lessons = lessons
        self._learn = learn
        self._learned: dict[str, None] = {}
        self._pending: set[asyncio.Task[None]] = set()

    async def record(self, incident: Incident, *, learn_key: str | None = None) -> None:
        """Write ``incident`` down, then maybe start the one rule update for its unit.

        ``learn_key`` identifies the unit of work. It defaults to the incident's
        mission, which is what the bus path wants; a chat delegation passes its
        own job id instead.
        """
        await asyncio.to_thread(self._lessons.record_incident, incident)
        key = learn_key if learn_key is not None else incident.mission_id
        self._maybe_learn(incident, key)

    def _label(self, incident: Incident) -> str:
        return incident.mission_id or incident.executor or "unknown"

    async def _run_learn(self, incident: Incident) -> None:
        try:
            raw = await self._learn(self._lessons.lesson_prompt(incident))  # type: ignore[misc]
            if not raw:
                logger.info("lessons: no subscription answered, incident %s waits",
                            self._label(incident))
                return
            applied = await asyncio.to_thread(self._lessons.apply, raw, incident.lane)
            logger.info("lessons: %d rule operation(s) from %s", applied, self._label(incident))
        except Exception:  # noqa: BLE001 - a lesson is optional, the incident is on disk
            logger.warning("lessons: rule update failed for %s", self._label(incident),
                           exc_info=True)

    def _maybe_learn(self, incident: Incident, key: str) -> None:
        if self._learn is None or key in self._learned:
            return
        if not key:
            # Without a key the same unit would buy a review per attempt, so the
            # rule update is skipped rather than fired blind. Said out loud,
            # because a silent skip reads like a dead seat (AP-30).
            logger.info("lessons: no learn key for incident %r, rule update skipped",
                        incident.cause[:80])
            return
        self._learned[key] = None
        while len(self._learned) > _LEARNED_MEMORY:
            self._learned.pop(next(iter(self._learned)))
        task = asyncio.get_running_loop().create_task(self._run_learn(incident))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


#: The recorder the running app documents into; see :func:`active_recorder`.
#: Set by :mod:`jarvis.missions.init` only -- never by
#: :func:`bind_incident_recorder`, so a test that binds a recorder to a tmp
#: store cannot leak it into the next test.
_active: IncidentRecorder | None = None


def set_active_recorder(recorder: IncidentRecorder | None) -> None:
    """Publish the process-wide recorder (mirrors ``lessons.set_active_store``)."""
    global _active
    _active = recorder


def active_recorder() -> IncidentRecorder | None:
    """The recorder a non-mission caller documents into, or None before boot."""
    return _active


def bind_incident_recorder(
    bus: Any,
    lessons: LessonStore,
    prompt_of: PromptLookup,
    learn: LessonAsk | None = None,
) -> IncidentRecorder:
    """Record an incident for every non-approving verdict and every failed mission.

    With ``learn``, the first incident of each mission also updates the rules.
    Returns the recorder so the caller can publish it with
    :func:`set_active_recorder` for the non-mission callers.
    """
    recorder = IncidentRecorder(lessons, learn)

    async def _handler(env: EventEnvelope) -> None:
        payload = env.payload
        if isinstance(payload, CriticVerdictReady) and payload.verdict != "approve":
            cause = f"critic {payload.verdict} (iter {payload.iteration}): {payload.summary}"
            executor = payload.reviewer_family or ""
        elif isinstance(payload, MissionFailed):
            detail = f": {payload.error_detail}" if payload.error_detail else ""
            cause = f"mission failed ({payload.reason}){detail}"
            executor = payload.failed_provider or ""
        else:
            return
        try:
            request = clean_request_body(await prompt_of(env.mission_id) or "")
        except Exception:  # noqa: BLE001 - the incident is still worth keeping
            logger.warning("incidents: prompt lookup failed for %s", env.mission_id,
                           exc_info=True)
            request = ""
        incident = Incident(
            lane=first_lane(request) if request else "unknown",
            executor=executor, cause=cause, request=request[:500],
            mission_id=env.mission_id,
        )
        await recorder.record(incident, learn_key=env.mission_id)

    bus.subscribe_all(_handler)
    return recorder


__all__ = [
    "IncidentRecorder", "LessonAsk", "active_recorder", "bind_incident_recorder",
    "set_active_recorder",
]
