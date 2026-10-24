"""Write every mission failure down as it happens, without a model call.

Subscribes to the mission bus like the budget tracker: a critic verdict that
is not an approval and a failed mission each become one
:class:`.lessons.Incident`. The record itself never needs a model, so nothing
is lost when every provider is down.

With a ``learn`` callable, the first incident of a mission also starts ONE rule
update (:meth:`.lessons.LessonStore.lesson_prompt` -> ``learn`` ->
:meth:`.lessons.LessonStore.apply`) in the background. ``learn`` is the
conductor's subscription seat; it returns None when no subscription can answer,
and the incident then simply waits on disk (background policy: never a key).
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

#: Missions remembered as "lesson already asked", so a revise loop costs one call.
_LEARNED_MEMORY = 256


def bind_incident_recorder(
    bus: Any,
    lessons: LessonStore,
    prompt_of: PromptLookup,
    learn: LessonAsk | None = None,
) -> None:
    """Record an incident for every non-approving verdict and every failed mission.

    With ``learn``, the first incident of each mission also updates the rules.
    """
    learned: dict[str, None] = {}
    pending: set[asyncio.Task[None]] = set()

    async def _learn(incident: Incident) -> None:
        try:
            raw = await learn(lessons.lesson_prompt(incident))  # type: ignore[misc]
            if not raw:
                logger.info("lessons: no subscription answered, incident %s waits",
                            incident.mission_id)
                return
            applied = await asyncio.to_thread(lessons.apply, raw, incident.lane)
            logger.info("lessons: %d rule operation(s) from %s", applied, incident.mission_id)
        except Exception:  # noqa: BLE001 - a lesson is optional, the incident is on disk
            logger.warning("lessons: rule update failed for %s", incident.mission_id,
                           exc_info=True)

    def _maybe_learn(incident: Incident) -> None:
        if learn is None or incident.mission_id in learned:
            return
        learned[incident.mission_id] = None
        while len(learned) > _LEARNED_MEMORY:
            learned.pop(next(iter(learned)))
        task = asyncio.get_running_loop().create_task(_learn(incident))
        pending.add(task)
        task.add_done_callback(pending.discard)

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
        await asyncio.to_thread(lessons.record_incident, incident)
        _maybe_learn(incident)

    bus.subscribe_all(_handler)


__all__ = ["LessonAsk", "bind_incident_recorder"]
