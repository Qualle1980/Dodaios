"""Stage A: live-path input transcripts and abort causes must be log-visible.

A 2026-11-13 live call showed garbled fragments of Jarvis' own reply
("RTTP", "tipica", "die") as chat "user" lines while the logfile had no
matching input-transcript entry — so the echo hypothesis could not be
decided from diagnostics alone. These pins require the native handler to
emit one structured log line per input transcript and per abort edge.
"""

from __future__ import annotations

import logging

import pytest

from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import gemini_like_provider

from .test_native_live_fixes import _in_call, _settle


@pytest.mark.asyncio
async def test_input_transcript_is_logged_with_output_state(tmp_path, caplog) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    session._speaking = True
    session.playback_active = False
    with caplog.at_level(logging.INFO, logger="jarvis.live.native"):
        await session._native_event(
            RealtimeEvent(type="input_transcript", text="RTTP", is_final=True)
        )
        await _settle(session)
    assert "live.native[native-live] input_transcript chars=4 final=True" in caplog.text
    assert "output_running=True" in caplog.text
    assert "text='RTTP'" in caplog.text


@pytest.mark.asyncio
async def test_interrupted_abort_logs_its_cause(tmp_path, caplog) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    session._speaking = True
    with caplog.at_level(logging.INFO, logger="jarvis.live.native"):
        await session._native_event(
            RealtimeEvent(type="interrupted", superseded=False, self_initiated=False)
        )
        await _settle(session)
    assert "live.native[native-live] abort cause=provider_interrupted" in caplog.text
    assert "output_running=True" in caplog.text


@pytest.mark.asyncio
async def test_superseded_interrupt_is_named_in_the_log(tmp_path, caplog) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    with caplog.at_level(logging.INFO, logger="jarvis.live.native"):
        await session._native_event(
            RealtimeEvent(type="interrupted", superseded=True, self_initiated=False)
        )
        await _settle(session)
    assert "abort cause=provider_superseded_regen" in caplog.text
