"""Delegation fallback for a brain that cannot emit ``tool_calls``.

Covers ``BrainManager._route_text_only_delegation``: the router of last resort
that runs only when no tool-capable provider could lead the turn. The method is
exercised unbound on a stub, so the test needs no real BrainManager (and no
brain, bridge or config).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.voice.action_phrases import action_phrase

_route = BrainManager._route_text_only_delegation


class _Stub:
    _active_name = "chatgpt-web"


class _Manager:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.languages: list[str] = []

    async def dispatch(self, *, prompt: str, language: str, source_actor: str) -> str:
        self.prompts.append(prompt)
        self.languages.append(language)
        return "mission-1"


class _Kontrollierer:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def run_mission(self, mission_id: str) -> None:
        self.ran.append(mission_id)


def _patch_stack(monkeypatch: pytest.MonkeyPatch, manager: Any, kontrollierer: Any) -> None:
    import jarvis.brain.factory as factory

    monkeypatch.setattr(factory, "_resolve_mission_manager", lambda: manager)
    monkeypatch.setattr(factory, "_resolve_kontrollierer", lambda: kontrollierer)


async def _settle(kontrollierer: _Kontrollierer) -> None:
    for _ in range(50):
        await asyncio.sleep(0)
        if kontrollierer.ran:
            return


@pytest.mark.asyncio
async def test_prose_reply_is_left_untouched() -> None:
    """``self`` means "this is the answer": the turn must not be rewritten."""
    out = await _route(
        _Stub(), "Das ist eine normale Antwort.", "hi", language="de", may_delegate=True
    )
    assert out is None


@pytest.mark.asyncio
async def test_clarify_returns_the_question() -> None:
    reply = '{"route": "clarify", "text": "Welche Datei meinst du?"}'
    out = await _route(_Stub(), reply, "mach das", language="de", may_delegate=True)
    assert out == "Welche Datei meinst du?"


@pytest.mark.asyncio
async def test_delegate_dispatches_and_runs_the_mission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both steps are mandatory: dispatch alone leaves the mission PENDING."""
    manager, kontrollierer = _Manager(), _Kontrollierer()
    _patch_stack(monkeypatch, manager, kontrollierer)

    reply = '{"route": "delegate", "target": "codex", "task": "Schreibe die Tests."}'
    spoken = await _route(_Stub(), reply, "schreib mir tests", language="de", may_delegate=True)

    assert spoken == action_phrase("delegate_task_started", "de")
    # The mission runs in a background task, so let it finish.
    for _ in range(10):
        await asyncio.sleep(0)
        if kontrollierer.ran:
            break
    assert kontrollierer.ran == ["mission-1"]
    assert manager.languages == ["de"]
    prompt = manager.prompts[0]
    assert "Schreibe die Tests." in prompt
    # The target travels as the payload's context, like spawn_worker does it.
    assert "Zielort/Kontext: codex." in prompt
    # The worker must never lose the user's own wording.
    assert "schreib mir tests" in prompt


@pytest.mark.asyncio
async def test_broken_envelope_never_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, kontrollierer = _Manager(), _Kontrollierer()
    _patch_stack(monkeypatch, manager, kontrollierer)

    for reply in (
        '{"route": "delegate", "target": "gemini", "task": "x"}',  # unknown target
        '{"route": "delegate", "target": "codex"}',  # no task
        '{"route": "delegate", "target": "codex", "task": "  "}',  # empty task
        '{"route": "nonsense", "text": "x"}',  # unknown route
        '{"route": "delegate", "target": "codex", "task": "x"',  # broken JSON
    ):
        assert await _route(_Stub(), reply, "u", language="de", may_delegate=True) is None
    await asyncio.sleep(0)
    assert manager.prompts == []
    assert kontrollierer.ran == []


@pytest.mark.asyncio
async def test_no_mission_stack_answers_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never promise work nobody will run (the bootstrap may not be done)."""
    _patch_stack(monkeypatch, None, _Kontrollierer())
    reply = '{"route": "delegate", "target": "claude", "task": "Etwas tun."}'
    assert await _route(_Stub(), reply, "u", language="de", may_delegate=True) is None


@pytest.mark.asyncio
async def test_envelope_is_unwrapped_when_delegation_is_not_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-capable provider led the turn: no second mission, but also no raw
    JSON in front of the user."""
    manager, kontrollierer = _Manager(), _Kontrollierer()
    _patch_stack(monkeypatch, manager, kontrollierer)

    reply = '{"route": "delegate", "target": "codex", "task": "Schreibe die Tests."}'
    out = await _route(_Stub(), reply, "u", language="de", may_delegate=False)

    assert out == "Schreibe die Tests."
    await asyncio.sleep(0)
    assert manager.prompts == []
    assert kontrollierer.ran == []


@pytest.mark.asyncio
async def test_an_informational_turn_keeps_the_question_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delegated question must stay recognisable as informational, or the
    critic can never approve it (it has no diff to grade)."""
    from jarvis.missions.stream_evidence import is_informational_request

    manager, kontrollierer = _Manager(), _Kontrollierer()
    _patch_stack(monkeypatch, manager, kontrollierer)

    reply = (
        '{"route": "delegate", "target": "codex", '
        '"task": "Nenne die erste Ueberschrift aus README.md. Aendere keine Datei."}'
    )
    await _route(
        _Stub(), reply, "was steht oben in der readme", language="de", may_delegate=True
    )
    await _settle(kontrollierer)

    prompt = manager.prompts[0]
    assert is_informational_request(prompt)
    assert prompt.startswith("was steht oben in der readme")
    assert "Zielort/Kontext: codex." in prompt
    assert not prompt.startswith("Aufgabe:")


@pytest.mark.asyncio
async def test_a_work_turn_keeps_the_task_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.missions.stream_evidence import is_informational_request

    manager, kontrollierer = _Manager(), _Kontrollierer()
    _patch_stack(monkeypatch, manager, kontrollierer)

    reply = (
        '{"route": "delegate", "target": "codex", '
        '"task": "Fuege der Datei bar.py eine Funktion foo() hinzu."}'
    )
    await _route(
        _Stub(), reply, "bau mir die funktion foo in bar.py", language="de", may_delegate=True
    )
    await _settle(kontrollierer)

    prompt = manager.prompts[0]
    assert prompt.startswith("Aufgabe: Fuege der Datei bar.py")
    assert "Urspruengliche Nutzeranfrage: bau mir die funktion foo in bar.py" in prompt
    assert not is_informational_request(prompt)
