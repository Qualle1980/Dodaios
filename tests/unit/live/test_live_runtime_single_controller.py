"""One voice session and one audio controller per conversation. No provider, no audio.

The registry in :mod:`jarvis.live.runtime` is what keeps a second call, a wake
word and the classic pipeline from grabbing the microphone at the same time.
These cases pin that contract so a later change cannot quietly allow two
controllers in one conversation.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jarvis.live import runtime


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every case starts and leaves with an empty registry."""
    runtime._active.clear()
    runtime._owners.clear()
    runtime._opening.clear()
    yield
    runtime._active.clear()
    runtime._owners.clear()
    runtime._opening.clear()


def _session(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(session_id=session_id)


def test_a_second_session_cannot_claim_while_one_is_opening() -> None:
    runtime.claim("first")

    with pytest.raises(RuntimeError):
        runtime.claim("second")


def test_a_second_session_cannot_claim_while_one_is_active() -> None:
    async def scenario() -> None:
        runtime.claim("first")
        runtime.register(_session("first"))

        with pytest.raises(RuntimeError):
            runtime.claim("second")

    asyncio.run(scenario())

    assert len(runtime.active()) == 1


def test_an_opening_call_already_owns_the_microphone() -> None:
    """A wake word must not replace a call that is still negotiating."""
    runtime.claim("first")

    assert runtime.owns_microphone() is True
    # The call does not block itself.
    assert runtime.owns_microphone(except_session_id="first") is False


def test_releasing_the_session_frees_the_microphone_and_allows_the_next() -> None:
    async def scenario() -> None:
        runtime.claim("first")
        runtime.register(_session("first"))
        assert runtime.owns_microphone() is True

        runtime.unregister("first")

        assert runtime.active() == ()
        assert runtime.owns_microphone() is False
        # The pipeline, or the next call, may take the device now.
        runtime.claim("second")

    asyncio.run(scenario())


def test_cleanup_is_idempotent_after_a_failed_start() -> None:
    """A call that never registered must not leave the microphone claimed."""
    runtime.claim("never-registered")
    runtime.unregister("never-registered")
    runtime.unregister("never-registered")

    assert runtime.owns_microphone() is False
    assert runtime.active() == ()


def test_live_tools_prefer_the_voice_catalog_so_project_tools_stay_connected() -> None:
    """The live call must see the project-scoped read tools, not the typed set.

    jarvis/live/tools.py reads ``voice_catalog`` when the gateway offers one and
    executes with ``origin="realtime"``; both together are what keeps the
    project files reachable in a call through the existing executor.
    """
    from pathlib import Path

    from jarvis.live import tools as live_tools

    source = Path(live_tools.__file__).read_text(encoding="utf-8")

    assert 'getattr(self.gateway, "voice_catalog", self.gateway.catalog)' in source
    assert 'origin="realtime"' in source
