"""The delegation hook inside the real ``BrainManager.generate()`` turn.

The helper's own behaviour is covered in ``test_text_only_delegation.py``; this
file checks the wiring: a text-only brain whose reply IS a routing envelope must
leave ``generate()`` as the spoken promise (never as raw JSON), the mission must
really be dispatched AND run, and an ordinary turn must not touch the mission
stack at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.brain.streaming import StreamingAggregate
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import BrainDelta, BrainRequest
from jarvis.voice.action_phrases import action_phrase

_ENVELOPE = '{"route": "delegate", "target": "codex", "task": "Schreibe die Tests."}'


class _TextOnlyBrain:
    """Mirrors ChatGPTWebBrain's capability surface: no tool calls at all."""

    name = "chatgpt-web"
    context_window = 128_000
    supports_tools = False
    supports_vision = False

    def can_call_tools(self) -> bool:
        return False

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        yield BrainDelta(content="ok")
        yield BrainDelta(finish_reason="stop")


class _Dispatcher:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def dispatch(self, user_text: str, **_kwargs: Any) -> StreamingAggregate:
        agg = StreamingAggregate()
        agg.text = self._reply
        agg.finish_reason = "stop"
        return agg


class _MissionManager:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def dispatch(self, *, prompt: str, language: str, source_actor: str) -> str:
        self.prompts.append(prompt)
        return "mission-live"


class _Kontrollierer:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def run_mission(self, mission_id: str) -> None:
        self.ran.append(mission_id)


def _manager(reply: str) -> BrainManager:
    cfg = JarvisConfig()
    cfg.brain.primary = "chatgpt-web"
    manager = BrainManager(config=cfg, bus=EventBus(), tools={})
    manager._build_fallback_chain = (  # type: ignore[method-assign]
        lambda _level: [("chatgpt-web", "chatgpt-web/instant")]
    )
    manager._get_brain = (  # type: ignore[method-assign]
        lambda _name, _model: _TextOnlyBrain()
    )
    manager._build_dispatcher = (  # type: ignore[method-assign]
        lambda _brain, **_kwargs: _Dispatcher(reply)
    )
    return manager


def _patch_stack(
    monkeypatch: pytest.MonkeyPatch, mission: Any, kontrollierer: Any
) -> None:
    import jarvis.brain.factory as factory

    monkeypatch.setattr(factory, "_resolve_mission_manager", lambda: mission)
    monkeypatch.setattr(factory, "_resolve_kontrollierer", lambda: kontrollierer)


async def _settle(kontrollierer: _Kontrollierer) -> None:
    for _ in range(50):
        await asyncio.sleep(0)
        if kontrollierer.ran:
            return


@pytest.mark.asyncio
async def test_generate_turns_an_envelope_into_a_running_mission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mission, kontrollierer = _MissionManager(), _Kontrollierer()
    _patch_stack(monkeypatch, mission, kontrollierer)

    answer = await _manager(_ENVELOPE).generate("schreib mir die tests")

    # The user hears a promise, never the JSON.
    assert "{" not in answer
    assert answer == action_phrase("delegate_task_started", "de")
    await _settle(kontrollierer)
    # Both steps ran: dispatch alone would leave the mission PENDING (BUG-016).
    assert kontrollierer.ran == ["mission-live"]
    assert "Schreibe die Tests." in mission.prompts[0]
    assert "Zielort/Kontext: codex." in mission.prompts[0]


@pytest.mark.asyncio
async def test_generate_leaves_an_ordinary_turn_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mission, kontrollierer = _MissionManager(), _Kontrollierer()
    _patch_stack(monkeypatch, mission, kontrollierer)

    answer = await _manager("Berlin ist die Hauptstadt.").generate("was ist die hauptstadt")

    assert answer == "Berlin ist die Hauptstadt."
    await asyncio.sleep(0)
    assert mission.prompts == []
    assert kontrollierer.ran == []
