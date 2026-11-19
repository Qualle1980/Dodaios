"""Typed chat fallback uses every reachable stand-in the person already set up.

A typed turn is work the person started, so a failed pick may move onto another
reachable account — including API-key cards and a connected subscription seat
that is not on the resolver chain. Background work stays on the stricter
no-surprise-invoice rule. Locals stay last. The turn names who answered.
"""

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
KEYED = "openrouter"
KEYED_MODEL = "keyed-model"
# Ollama is never an automatic successor (Ahmad 2026-11-17); local-openai stays.
LOCAL = "local-openai"
LOCAL_MODEL = "local-model"
SEAT = "claude-cli"
SEAT_MODEL = "claude-sonnet"
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


def _manager(
    bus: EventBus,
) -> tuple[BrainManager, _FailingBrain, FakeBrain, FakeBrain]:
    config = JarvisConfig()
    config.brain.primary = PICK
    config.brain.worker = BrainTierConfig(provider=KEYED, model=KEYED_MODEL)
    config.brain.local_fallback = LOCAL
    config.brain.local_fallback_model = LOCAL_MODEL
    config.brain.providers[PICK] = BrainProviderConfig(
        model=PICK_MODEL, deep_model=PICK_MODEL
    )
    manager = BrainManager(config=config, bus=bus, tools={})
    manager._registry._loaded = True

    picked = _FailingBrain(PICK_ERROR)
    keyed = FakeBrain(text_response="The keyed stand-in answered.")
    local = FakeBrain(text_response="The local runtime answered.")
    manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    manager._brain_cache[(f"{KEYED}@agent", KEYED_MODEL)] = keyed
    manager._brain_cache[(f"{LOCAL}@agent", LOCAL_MODEL)] = local
    return manager, picked, keyed, local


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
    keyed_fails: bool = False,
    local_fails: bool = False,
    pick_fails_midstream: bool = False,
    pick_reports_failed_stream: bool = False,
    seat_connected: bool = False,
    omit_keyed_from_chain: bool = False,
) -> tuple[list[dict[str, Any]], _FailingBrain, FakeBrain, FakeBrain, FakeBrain | None]:
    from jarvis.agent_chat import service as service_module
    from jarvis.brain import background_policy

    background_policy.reset_for_tests()
    # Machine-independent sign-in: only the seat under test is connected.
    # Never read the real CLI login state on this PC (order 2026-11-17).
    login_map = {SEAT: True} if seat_connected else {}
    background_policy._probe_override = lambda provider: login_map.get(provider, False)
    monkeypatch.setattr(background_policy, "_read_markers", lambda: {})
    monkeypatch.setattr(background_policy, "_write_markers", lambda _markers: None)
    monkeypatch.setattr(
        background_policy,
        "_registry_probe",
        lambda provider: login_map.get(provider, False),
    )

    bus = EventBus()
    manager, picked, keyed, local = _manager(bus)
    if omit_keyed_from_chain:
        manager._config.brain.worker = BrainTierConfig(provider="", model="")
    if pick_fails_midstream:
        picked = _PartialFailingBrain(PICK_ERROR)
        manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    if pick_reports_failed_stream:
        picked = _MarkedFailureBrain(PICK_ERROR)
        manager._brain_cache[(f"{PICK}@agent", PICK_MODEL)] = picked
    if keyed_fails:
        failed_keyed = _FailingBrain("keyed stand-in is offline")
        manager._brain_cache[(f"{KEYED}@agent", KEYED_MODEL)] = failed_keyed
        keyed = failed_keyed
    if local_fails:
        failed_local = _FailingBrain("local runtime is offline")
        manager._brain_cache[(f"{LOCAL}@agent", LOCAL_MODEL)] = failed_local
        local = failed_local

    seat: FakeBrain | None = None
    if seat_connected:
        seat = FakeBrain(text_response="The connected subscription answered.")
        manager._brain_cache[(f"{SEAT}@agent", SEAT_MODEL)] = seat
        manager._brain_cache[(f"{SEAT}@agent", None)] = seat
        manager._config.brain.providers[SEAT] = BrainProviderConfig(
            model=SEAT_MODEL, deep_model=SEAT_MODEL
        )

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
        return events, picked, keyed, local, seat
    finally:
        service.unsubscribe(session.session_id, queue)
        service.store.close()
        background_policy.reset_for_tests()


@pytest.mark.asyncio
async def test_a_failed_chat_pick_is_answered_by_a_keyed_stand_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, keyed, local, _seat = await _typed_turn(tmp_path, monkeypatch)

    assert picked.calls
    assert keyed.calls
    assert local.calls == []
    assert next(e for e in events if e["kind"] == "assistant_text")["payload"]["text"] == (
        "The keyed stand-in answered."
    )
    assert events[-1]["payload"]["status"] == "done"


@pytest.mark.asyncio
async def test_the_visible_turn_receipt_names_the_keyed_stand_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, _picked, _keyed, _local, _seat = await _typed_turn(tmp_path, monkeypatch)

    assert events[-1]["payload"]["answered_by"] == {
        "provider": KEYED,
        "model": KEYED_MODEL,
    }


@pytest.mark.asyncio
async def test_a_local_runtime_is_only_tried_after_every_cloud_stand_in_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, keyed, local, _seat = await _typed_turn(
        tmp_path, monkeypatch, keyed_fails=True
    )

    assert picked.calls
    assert keyed.calls
    assert local.calls
    assert next(e for e in events if e["kind"] == "assistant_text")["payload"]["text"] == (
        "The local runtime answered."
    )
    assert events[-1]["payload"]["answered_by"] == {
        "provider": LOCAL,
        "model": LOCAL_MODEL,
    }


@pytest.mark.asyncio
async def test_a_connected_subscription_stands_in_even_outside_the_resolver_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, keyed, local, seat = await _typed_turn(
        tmp_path,
        monkeypatch,
        seat_connected=True,
        omit_keyed_from_chain=True,
        keyed_fails=True,
    )

    assert seat is not None
    assert picked.calls
    assert seat.calls
    assert local.calls == []
    assert next(e for e in events if e["kind"] == "assistant_text")["payload"]["text"] == (
        "The connected subscription answered."
    )
    assert events[-1]["payload"]["answered_by"]["provider"] == SEAT


def test_background_work_still_refuses_a_keyed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.brain import background_policy

    background_policy.reset_for_tests()
    background_policy._probe_override = lambda provider: provider == SEAT
    monkeypatch.setattr(background_policy, "_read_markers", lambda: {})
    monkeypatch.setattr(background_policy, "_write_markers", lambda _markers: None)

    decision = background_policy.background_providers(
        [SEAT, KEYED, "gemini", LOCAL]
    )

    assert decision.subscription_mode
    assert KEYED not in decision.allowed
    assert "gemini" not in decision.allowed
    assert SEAT in decision.allowed
    assert LOCAL in decision.allowed
    background_policy.reset_for_tests()


@pytest.mark.asyncio
async def test_no_representative_reports_the_pick_and_its_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, _picked, _keyed, _local, _seat = await _typed_turn(
        tmp_path, monkeypatch, keyed_fails=True, local_fails=True
    )

    terminal = events[-1]["payload"]
    assert terminal["status"] == "error"
    assert PICK in terminal["error"]
    assert PICK_ERROR in terminal["error"]


@pytest.mark.asyncio
async def test_a_midstream_failure_never_splices_a_second_provider_into_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, keyed, local, _seat = await _typed_turn(
        tmp_path, monkeypatch, pick_fails_midstream=True
    )

    assert picked.calls
    assert keyed.calls == []
    assert local.calls == []
    assert events[-1]["payload"]["status"] == "error"
    assert PICK_ERROR in events[-1]["payload"]["error"]
    assert not any(
        e["kind"] == "assistant_text"
        and "keyed stand-in" in e["payload"].get("text", "")
        for e in events
    )


@pytest.mark.asyncio
async def test_a_provider_error_finish_reason_is_not_reported_as_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, picked, keyed, local, _seat = await _typed_turn(
        tmp_path, monkeypatch, pick_reports_failed_stream=True
    )

    assert picked.calls
    assert keyed.calls == []
    assert local.calls == []
    assert events[-1]["payload"]["status"] == "error"
