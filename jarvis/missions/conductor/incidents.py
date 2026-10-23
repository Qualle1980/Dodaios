"""Write every mission failure down as it happens, without a model call.

Subscribes to the mission bus like the budget tracker: a critic verdict that
is not an approval and a failed mission each become one
:class:`.lessons.Incident`. The rule update that turns incidents into lessons
runs later on the conductor seat; this part only makes sure nothing is lost.
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


def bind_incident_recorder(bus: Any, lessons: LessonStore, prompt_of: PromptLookup) -> None:
    """Record an incident for every non-approving verdict and every failed mission."""

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
        await asyncio.to_thread(lessons.record_incident, Incident(
            lane=first_lane(request) if request else "unknown",
            executor=executor, cause=cause, request=request[:500],
            mission_id=env.mission_id,
        ))

    bus.subscribe_all(_handler)


__all__ = ["bind_incident_recorder"]
