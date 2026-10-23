"""Naming the coding agent that should do the work is an explicit request.

Live 2026-10-23: "Lass Codex die TODO-Kommentare in jarvis/missions/budget.py
auflisten" was logged as "balanced mode needs an explicit delegation
trigger", led by a tool provider that could not answer, and the seat replied
that it could not reach Codex. No mission started.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig


class _FakeSpawnTool:
    name = "spawn_worker"
    schema: dict[str, Any] = {}


class _InertExecutor:
    async def execute(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover
        raise AssertionError("executor must not run in a classification test")


def _manager(mode: str) -> BrainManager:
    config = JarvisConfig()
    config.brain.routing.force_spawn_mode = mode
    return BrainManager(
        config=config,
        bus=EventBus(),
        tools={"spawn_worker": _FakeSpawnTool()},
        tool_executor=_InertExecutor(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("mode", ["strict", "balanced"])
@pytest.mark.parametrize(
    "utterance",
    [
        "Lass Codex die TODO-Kommentare in jarvis/missions/budget.py auflisten.",  # i18n-allow
        "Frag Claude Code, warum der Test in test_routing.py rot ist.",  # i18n-allow
        "Ask Codex to list the TODOs in jarvis/missions/budget.py.",
        "Have Claude Code review the diff on this branch.",
    ],
)
def test_naming_the_vehicle_spawns(mode: str, utterance: str) -> None:
    assert _manager(mode)._should_force_spawn(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "Was ist Codex eigentlich?",  # i18n-allow
        "Is Claude Code better than Codex?",
    ],
)
def test_talking_about_the_vehicle_does_not_spawn(utterance: str) -> None:
    assert _manager("balanced")._should_force_spawn(utterance) is False
