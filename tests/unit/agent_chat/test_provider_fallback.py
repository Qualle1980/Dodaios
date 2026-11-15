"""Typed chat fallback stays useful without spending an unapproved API key."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent_chat import runner_brain
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainProviderConfig, BrainTierConfig, JarvisConfig
from jarvis.core.protocols import BrainDelta, BrainRequest
from tests.fixtures.brain.fake_brain import FakeBrain

PICK = "openai"
PICK_MODEL = "picked-model"
REPRESENTATIVE = "ollama"
REPRESENTATIVE_MODEL = "local-model"
FORBIDDEN = "openrouter"
FORBIDDEN_MODEL = "keyed-model"
PICK_ERROR = "selected account is temporarily unavailable"


class _FailingBrain(FakeBrain):
    """A provider fake that preserves the exact failure text."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    async def complete(self, req: BrainRequest):  # type: ignore[no-untyped-def]
        self.calls.append(req)
        raise RuntimeError(self.message)
        yield  # pragma: no cover - keeps this an async generator


class _PartialFailingBrain(_FailingBrain):
    """A failed attempt that already streamed words to the chat."""

    async def complete(self, req: BrainRequest):  # type: ignore[no-untyped-def]
        self.calls.append(req)
        yield BrainDelta(content="Incomplete pick response. ")
        raise RuntimeError(self.message)


class _MarkedFailureBrain(_FailingBrain):
    """A provider that reports a failed stream instead of raising."""

    async def complete(self, req: BrainRequest):  # type: ignore[no-untyped-def]
        self.calls.append(req)
        yield BrainDelta(content="Unfinished picked answer. ", finish_reason="error")


def _manager(bus: EventBus) -> tuple[BrainManager, _FailingBrain, FakeBrain, FakeBrain]:
    config = JarvisConfig()
    config.brain.primary = PICK
    config.brain.worker = BrainTierConfig(provider=FORBIDDEN, model=FORBIDDEN_MODEL)
    config.brain.local_fallback = REPRESENTATIVE
    config.brain.local_fallback_model = REPRESENTATIVE_MODEL
    config.brain.providers[PICK] = BrainProviderConfig(
        model=PICK_MODEL, deep_model=PICK_MODEL
    )
    manager = BrainManager(config=config, bus=bus, tools={})
    manager._registry._loaded = True

    picked = _FailingBrain(PICK_ERROR)
    representative = FakeBrain(text_response="The safe fallback answered.")
    forbidden = FakeBrain(text_response="The API-key provider must never answer.")
    manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    manager._brain_cache[(f"{REPRESENTATIVE}@agent", REPRESENTATIVE_MODEL)] = representative
    manager._brain_cache[(f"{FORBIDDEN}@agent", FORBIDDEN_MODEL)] = forbidden
    return manager, picked, representative, forbidden


def _service(bus: EventBus) -> AgentChatService:
    return AgentChatService(
        AgentChatStore(":memory:"), assistant_name=lambda: "Testo", bus=lambda: bus
    )


async def _drain(queue: asyncio.Queue, until_kind: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with asyncio.timeout(5):
        while True:
            event = await queue.get()
            events.append(event)
            if event["kind"] == until_kind:
                return events


async def _typed_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    representative_fails: bool = False,
    pick_fails_midstream: bool = False,
    pick_reports_failed_stream: bool = False,
) -> tuple[list[dict[str, Any]], _FailingBrain, FakeBrain, FakeBrain]:
    from jarvis.agent_chat import service as service_module
    from jarvis.brain import background_policy

    background_policy.reset_for_tests()
    background_policy._probe_override = lambda _provider: False
    monkeypatch.setattr(background_policy, "_read_markers", lambda: {})
    monkeypatch.setattr(background_policy, "_write_markers", lambda _markers: None)

    bus = EventBus()
    manager, picked, representative, forbidden = _manager(bus)
    if pick_fails_midstream:
        picked = _PartialFailingBrain(PICK_ERROR)
        manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    if pick_reports_failed_stream:
        picked = _MarkedFailureBrain(PICK_ERROR)
        manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    if representative_fails:
        failed_local = _FailingBrain("local runtime is offline")
        manager._brain_cache[
            (f"{REPRESENTATIVE}@agent", REPRESENTATIVE_MODEL)
        ] = failed_local
        representative = failed_local

    monkeypatch.setattr(service_module, "_claude_cli_installed", lambda: False)
    monkeypatch.setattr(runner_brain, "brain_manager", lambda: manager)
    monkeypatch.setattr(runner_brain, "_agent_secret", lambda _resolver, _provider: None)

    service = _service(bus)
    session = service.create_session(
        provider=PICK,
        model=PICK_MODEL,
        effort="",
        cwd=str(tmp_path),
        permission_mode="ask",
        surface="jarvis",
    )
    queue = service.subscribe(session.session_id)
    try:
        await service.send(session.session_id, "Explain the fallback in one sentence.")
        events = await _drain(queue, "turn_finished")
        await service.wait_turn(session.session_id)
        return events, picked, representative, forbidden
    finally:
        service.unsubscribe(session.session_id, queue)
        service.store.close()
        background_policy.reset_for_tests()


@pytest.mark.asyncio
async def test_a_failed_chat_pick_is_answered_by_an_allowed_representative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, representative, _forbidden = await _typed_turn(tmp_path, monkeypatch)

    assert picked.calls
    assert representative.calls
    assert next(e for e in events if e["kind"] == "assistant_text")["payload"]["text"] == (
        "The safe fallback answered."
    )
    assert events[-1]["payload"]["status"] == "done"


@pytest.mark.asyncio
async def test_the_visible_turn_receipt_names_the_representative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, _picked, _representative, _forbidden = await _typed_turn(tmp_path, monkeypatch)

    assert events[-1]["payload"]["answered_by"] == {
        "provider": REPRESENTATIVE,
        "model": REPRESENTATIVE_MODEL,
    }


@pytest.mark.asyncio
async def test_no_representative_reports_the_pick_and_its_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, _picked, _representative, _forbidden = await _typed_turn(
        tmp_path, monkeypatch, representative_fails=True
    )

    terminal = events[-1]["payload"]
    assert terminal["status"] == "error"
    assert PICK in terminal["error"]
    assert PICK_ERROR in terminal["error"]


@pytest.mark.asyncio
async def test_a_keyed_provider_forbidden_by_policy_is_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _events, _picked, _representative, forbidden = await _typed_turn(tmp_path, monkeypatch)

    assert forbidden.calls == []


@pytest.mark.asyncio
async def test_a_midstream_failure_never_splices_a_second_provider_into_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, representative, forbidden = await _typed_turn(
        tmp_path, monkeypatch, pick_fails_midstream=True
    )

    assert picked.calls
    assert representative.calls == []
    assert forbidden.calls == []
    assert events[-1]["payload"]["status"] == "error"
    assert PICK_ERROR in events[-1]["payload"]["error"]
    assert not any(
        e["kind"] == "assistant_text"
        and "safe fallback" in e["payload"].get("text", "")
        for e in events
    )


@pytest.mark.asyncio
async def test_a_provider_error_finish_reason_is_not_reported_as_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, representative, forbidden = await _typed_turn(
        tmp_path, monkeypatch, pick_reports_failed_stream=True
    )

    assert picked.calls
    assert representative.calls == []
    assert forbidden.calls == []
    assert events[-1]["payload"]["status"] == "error"
