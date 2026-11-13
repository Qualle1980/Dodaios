"""Native live path must drop provider-transcribed speaker echo (BUG-089 shape).

The realtime session already registers spoken text with ``SelfEchoGuard`` and
correlates barge-in candidates against ``echo_reference``. The native live
path (``jarvis/live/native.py``) did neither: Gemini heard our own browser
playback through the mic, transcribed garble like "RTTP", and each fragment
opened a new user turn that cancelled the answer in flight.

These pins require the native handler to (1) record outbound PCM into the
echo envelope, (2) drop a final input transcript that is only our own recent
speech while that envelope is active, and (3) still accept a novel barge-in
during playback.
"""

from __future__ import annotations

import logging
import struct

import pytest

from jarvis.audio import echo_reference
from jarvis.core.protocols import AudioChunk
from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import gemini_like_provider

from .test_native_live_fixes import _in_call, _settle


def _loud_pcm(samples: int = 480, amplitude: int = 8_000) -> bytes:
    return struct.pack("<" + "h" * samples, *([amplitude] * samples))


def _quiet_pcm(samples: int = 480) -> bytes:
    return struct.pack("<" + "h" * samples, *([0] * samples))


@pytest.fixture(autouse=True)
def _clean_echo_reference():
    echo_reference.reset()
    yield
    echo_reference.reset()


@pytest.mark.asyncio
async def test_outbound_audio_feeds_echo_reference(tmp_path) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    # ``_in_call`` skips ``_start``, so the provider handle is installed here.
    session._provider = gemini_like_provider()
    pcm = _loud_pcm(4_800)  # 200 ms at 24 kHz
    await session._native_event(
        RealtimeEvent(
            type="audio_delta",
            audio=AudioChunk(pcm=pcm, sample_rate=24_000, timestamp_ns=0),
        )
    )
    entries = echo_reference.snapshot(5.0)
    assert entries, "native outbound PCM must land in echo_reference"
    assert sum(duration for _ts, duration, _rms in entries) == pytest.approx(
        0.2, abs=0.02
    )


@pytest.mark.asyncio
async def test_self_echo_transcript_is_dropped_during_playback(
    tmp_path, caplog
) -> None:
    session, connection, frames = _in_call(gemini_like_provider(), tmp_path)
    session._speaking = True
    session.playback_active = True
    echo_reference.record(0.2, 0.1)
    await session._native_event(
        RealtimeEvent(
            type="output_transcript_delta",
            text="Die Antwort steht bereit tipica",  # i18n-allow: echo fixture
        )
    )
    with caplog.at_level(logging.INFO, logger="jarvis.live.native"):
        await session._native_event(
            # Truncated speaker-echo fragment (BUG-101 short path).
            RealtimeEvent(type="input_transcript", text="tipica die", is_final=True)
        )
        await _settle(session)
    assert "dropped provider-transcribed self-echo" in caplog.text
    assert not any(
        frame.get("role") == "user"
        for frame in frames
        if frame.get("type") == "transcript"
    )
    assert "interrupt" in connection.names()
    assert session._captions.get("user") in (None, "")


@pytest.mark.asyncio
async def test_novel_barge_in_during_playback_still_becomes_a_turn(
    tmp_path,
) -> None:
    session, connection, frames = _in_call(gemini_like_provider(), tmp_path)
    session._speaking = True
    session.playback_active = True
    echo_reference.record(0.2, 0.1)
    await session._native_event(
        RealtimeEvent(
            type="output_transcript_delta",
            text="Die Antwort steht bereit tipica",  # i18n-allow: echo fixture
        )
    )
    await session._native_event(
        RealtimeEvent(
            type="input_transcript",
            text="stop please tell me the weather",
            is_final=True,
        )
    )
    await _settle(session)
    user_frames = [
        frame
        for frame in frames
        if frame.get("type") == "transcript" and frame.get("role") == "user"
    ]
    assert user_frames
    assert user_frames[0]["text"] == "stop please tell me the weather"
    assert session._captions.get("user") == "stop please tell me the weather"
    # A barge-in must not be mistaken for echo and cancelled by the drop path.
    assert connection.names().count("interrupt") == 0


@pytest.mark.asyncio
async def test_quiet_mic_is_held_while_echo_reference_is_active(tmp_path) -> None:
    session, connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    echo_reference.record(0.25, 0.1)
    await session.handle_audio_frame(_quiet_pcm())
    assert connection.names() == []


@pytest.mark.asyncio
async def test_loud_mic_still_reaches_provider_during_playback(tmp_path) -> None:
    from jarvis.realtime.audio import StreamingPcm16Resampler

    session, connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    session._provider = gemini_like_provider()
    session._resampler = StreamingPcm16Resampler(16_000, 16_000)
    echo_reference.record(0.05, 0.1)
    await session.handle_audio_frame(_loud_pcm(samples=320, amplitude=20_000))
    assert "send_audio" in connection.names()


@pytest.mark.asyncio
async def test_browser_media_levels_feed_echo_reference(tmp_path) -> None:
    session, _connection, _frames = _in_call(gemini_like_provider(), tmp_path)
    await session._receive_media_levels(
        {
            "type": "media_levels",
            "input_level": 0.0,
            "output_level": 0.4,
            "input_active": False,
            "playback_active": True,
        }
    )
    entries = echo_reference.snapshot(5.0)
    assert entries
    assert entries[-1][2] == pytest.approx(0.4)
    assert entries[-1][1] == pytest.approx(0.1)
