"""The native path records what it said, so the honesty check can read it.

#118 added a check that reports a spoken "I cannot do that" the attached
tools contradict. It logged nothing over four live calls carrying such
sentences, including 2026-11-07 22:05, where it denied access to the owner's
local Obsidian while wiki-recall sat in the surface. The detector is
fine -- fed that sentence and that surface it reports -- but it reads
``_captions["assistant"]``, and only the NON-native event handler ever wrote
that field. ``NativeLiveVoiceSession._native_event`` builds the same caption
and dropped it, so the check saw an empty string and returned.

Same shape as #111, which patched ``jarvis/realtime/session.py`` while the
app runs ``jarvis/live/``: right intent, a path the app does not take. Here
it was the right class and the wrong field.
"""

from __future__ import annotations

import logging

import pytest

from jarvis.core.protocols import SupervisorToolDescriptor
from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import FakeToolGateway, gemini_like_provider

from .test_native_live_fixes import _in_call, _settle


@pytest.mark.asyncio
async def test_what_the_call_said_is_recorded(tmp_path) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    await session._native_event(
        RealtimeEvent(type="output_transcript_delta", text="I checked the vault.", is_final=True)
    )
    await _settle(session)
    assert session._captions["assistant"] == "I checked the vault."


@pytest.mark.asyncio
async def test_what_the_user_said_is_recorded(tmp_path) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    await session._native_event(
        RealtimeEvent(type="input_transcript", text="look in my vault", is_final=True)
    )
    await _settle(session)
    assert session._captions["user"] == "look in my vault"


@pytest.mark.asyncio
async def test_a_contradicted_denial_now_reaches_the_log(tmp_path, caplog) -> None:
    """The end the whole check exists for: a false denial becomes a record."""
    vault_tool = SupervisorToolDescriptor(
        "wiki-recall",
        "Search the user's long-term Obsidian wiki for notes matching keywords.",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        "safe",
    )
    session, _connection, _frames = _in_call(
        gemini_like_provider(), tmp_path, gateway=FakeToolGateway((vault_tool,))
    )
    await session._native_event(
        RealtimeEvent(
            type="output_transcript_delta",
            text="Ich habe keinen direkten Zugriff auf dein lokales Obsidian.",  # i18n-allow: DE
            is_final=True,
        )
    )
    await _settle(session)
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert "capability denied over voice" in caplog.text
