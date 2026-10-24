"""A mission ordered in a typed chat writes its outcome back into that chat."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.events import DelegationResultReady
from jarvis.core.protocols import current_chat_turn
from jarvis.missions.events import (
    EventEnvelope,
    MissionApproved,
    MissionFailed,
    MissionWaitingCapacity,
)
from jarvis.missions.manager import MissionManager
from jarvis.plugins.tool.spawn_worker import SpawnWorkerTool, _typed_chat_session


class _Kontrollierer:
    """Ends the mission with one given terminal payload, like run_mission does."""

    def __init__(self, manager: MissionManager, payload: Any) -> None:
        self.manager = manager
        self.payload = payload

    async def run_mission(self, mission_id: str) -> None:
        await self.manager.store.append_and_publish(EventEnvelope(
            mission_id=mission_id, source_actor="kontrollierer", ts_ms=2, payload=self.payload))


@pytest.fixture
async def manager(tmp_path: Path):
    m = MissionManager(tmp_path / "missions.db")
    await m.start()
    yield m
    await m.stop()


async def _run(manager: MissionManager, payload: Any, *, chat: str, language: str = "en"):
    bus = EventBus()
    seen: list[DelegationResultReady] = []

    async def _keep(event: DelegationResultReady) -> None:
        seen.append(event)

    bus.subscribe(DelegationResultReady, _keep)
    tool = SpawnWorkerTool(bus=bus, manager=manager)
    await tool._background_dispatch("do it", "do it", manager, _Kontrollierer(manager, payload),
                                    mission_language=language, chat_session=chat)
    return seen


def _approved(**summaries: str) -> MissionApproved:
    return MissionApproved(result_uri="mission://x", tokens_used=0, cost_usd=0.0, wall_ms=0,
                           summary_de=summaries.get("de", ""), summary_en=summaries.get("en", ""))


async def test_an_approved_mission_posts_its_answer_in_the_dispatch_language(
    manager: MissionManager,
) -> None:
    seen = await _run(manager, _approved(de="Fertig.", en="Done: 234 lines."), chat="s1")  # i18n-allow: a German readback
    assert len(seen) == 1
    assert (seen[0].session_id, seen[0].status, seen[0].text) == ("s1", "done", "Done: 234 lines.")
    assert seen[0].request_id.startswith("mission:")

    seen = await _run(manager, _approved(de="Fertig.", en="Done."), chat="s1", language="de")  # i18n-allow
    assert seen[0].text == "Fertig."  # i18n-allow


async def test_a_failed_mission_posts_reason_and_detail(manager: MissionManager) -> None:
    seen = await _run(manager, MissionFailed(reason="conductor_stopped", last_state="RUNNING",
                                             error_detail="the file could not be written"),
                      chat="s2")
    assert seen[0].status == "failed"
    assert seen[0].text == "conductor_stopped: the file could not be written"


async def test_voice_orders_and_parked_missions_post_nothing(manager: MissionManager) -> None:
    assert await _run(manager, _approved(en="Done."), chat="") == []
    parked = MissionWaitingCapacity(reason="provider_quota", provider="codex")
    assert await _run(manager, parked, chat="s3") == []


def test_only_a_typed_chat_turn_names_its_session() -> None:
    from jarvis.core.chat_turn import ChatTurn

    assert _typed_chat_session() == ""
    token = current_chat_turn.set(ChatTurn(session_id="s9", turn_id="t", user_text="x",
                                            direct_user=True, trace_id="tr"))
    try:
        assert _typed_chat_session() == "s9"
    finally:
        current_chat_turn.reset(token)
