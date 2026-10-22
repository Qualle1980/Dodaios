"""The no-first-frame watchdog across a multi-sentence answer.

Live 2026-10-22 17:42:28: the turn's sentence player resets the player's
``last_write_ns`` to 0 at every sentence, and the watchdog re-armed its
no-first-frame window only on brain heartbeats. Once the brain had finished,
the earlier sentences' playback counted as silence, and the 5th sentence was
aborted 45 ms after it started — 20 s after the last brain token. Audio that
this playback really wrote now re-arms the window.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.protocols import AudioChunk
from jarvis.speech.pipeline import SpeechPipeline


@dataclass
class _NoTTS:
    name: str = "fake-tts"
    supports_streaming: bool = True

    async def synthesize(self, text, voice=None, language_code=None) -> AsyncIterator[AudioChunk]:
        if False:  # pragma: no cover - empty async generator
            yield


@dataclass
class SentencePlayer:
    """Plays sentences in one task, resetting progress per sentence like the real player.

    ``gaps_s[i]`` is the synthesis wait before sentence i's first frame,
    ``None`` means that sentence never yields a frame.
    """

    gaps_s: list[float | None]
    play_s: float = 0.5
    last_write_ns: int = 0
    played: list[int] = field(default_factory=list)
    aborted: bool = False
    _release: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self) -> None:
        for i, gap in enumerate(self.gaps_s):
            self.last_write_ns = 0  # play_chunks' per-sentence reset
            if gap is None:
                await self._release.wait()
                return
            await asyncio.sleep(gap)
            end = time.monotonic() + self.play_s
            while time.monotonic() < end:
                if self.aborted:
                    return
                await asyncio.sleep(0.02)
                self.last_write_ns = time.monotonic_ns()
            self.played.append(i)

    def abort_active(self) -> None:
        self.aborted = True
        self._release.set()

    def stop(self) -> None:
        self.abort_active()


def _pipeline(player: SentencePlayer, ceiling: float) -> SpeechPipeline:
    pipeline = SpeechPipeline(tts=_NoTTS(), bus=EventBus(), enable_whisper_wake=False)
    pipeline._player = player  # type: ignore[assignment]
    pipeline._speak_playback_ceiling_s = ceiling  # type: ignore[attr-defined]
    pipeline._speak_playback_stall_s = 5.0  # type: ignore[attr-defined]
    return pipeline


@pytest.mark.asyncio
async def test_consecutive_sentences_longer_than_ceiling_all_play() -> None:
    # 5 sentences x 0.5 s ≈ 2.7 s of audio against a 0.6 s ceiling, no brain
    # heartbeat at all (the brain already finished). Before the fix the
    # watchdog aborted at the first sentence boundary after 0.6 s.
    player = SentencePlayer(gaps_s=[0.05] * 5)
    pipeline = _pipeline(player, ceiling=0.6)
    play_task = asyncio.create_task(player.run())
    done = await asyncio.wait_for(pipeline._await_playback(play_task, set()), timeout=5)
    assert done == {play_task}
    assert player.played == [0, 1, 2, 3, 4], "the last sentence must not be lost"
    assert not player.aborted
    assert not getattr(pipeline, "_playback_aborted_no_first_frame", False)


@pytest.mark.asyncio
async def test_slow_sentence_within_ceiling_is_waited_for() -> None:
    # The 3rd sentence synthesizes for 0.4 s (< 0.6 s ceiling) after 1.1 s
    # of earlier playback: slow, but alive.
    player = SentencePlayer(gaps_s=[0.05, 0.05, 0.4, 0.05])
    pipeline = _pipeline(player, ceiling=0.6)
    play_task = asyncio.create_task(player.run())
    done = await asyncio.wait_for(pipeline._await_playback(play_task, set()), timeout=5)
    assert done == {play_task}
    assert player.played == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_sentence_that_never_starts_is_aborted_one_ceiling_after_last_audio() -> None:
    # Sentence 3 never yields a frame: the abort must still come, measured
    # from the last real audio — not from the start of the turn.
    player = SentencePlayer(gaps_s=[0.05, 0.05, None])
    pipeline = _pipeline(player, ceiling=0.6)
    play_task = asyncio.create_task(player.run())
    t0 = time.monotonic()
    done = await asyncio.wait_for(pipeline._await_playback(play_task, set()), timeout=5)
    elapsed = time.monotonic() - t0
    assert done == set()
    assert player.aborted
    assert pipeline._playback_aborted_no_first_frame is True  # type: ignore[attr-defined]
    assert player.played == [0, 1]
    # ~1.1 s of audio + 0.6 s ceiling (+ 0.25 s poll), never the 5 s guard.
    assert 1.6 <= elapsed < 2.5, elapsed
    await asyncio.wait_for(play_task, timeout=1)  # released, nothing left running


def test_output_rate_prefers_a_rate_the_device_already_accepted(monkeypatch) -> None:
    import jarvis.audio.player as player_mod
    from jarvis.audio.player import AudioPlayer

    opened: list[int] = []

    class _Stream:
        latency = 0.2

        def start(self) -> None:
            pass

    def _output_stream(*, samplerate, device, **_kw):
        opened.append(samplerate)
        if samplerate == 48000:  # device 11 only takes 48 kHz
            return _Stream()
        raise player_mod.sd.PortAudioError(
            "Error opening OutputStream: Invalid sample rate [PaErrorCode -9997]"
        )

    monkeypatch.setattr(player_mod.sd, "OutputStream", _output_stream)
    monkeypatch.setattr(player_mod.sd, "query_devices", lambda d: {"default_samplerate": 48000})
    p = AudioPlayer.__new__(AudioPlayer)
    p._device, p._sample_rate, p._channels = 11, 24_000, 1
    p._device_logged, p._bus, p._play_lock = True, None, None
    p._active_stream = p._active_source_rate = p._active_device_rate = None
    p._device_rate_cache, p._device_rate_failed = {}, set()

    assert p._open_output_stream(24_000)[1] == 48000
    opened.clear()
    # The 44.1 kHz voice: before, 44100 was tried and refused first.
    assert p._open_output_stream(44_100)[1] == 48000
    assert opened == [48000]


def test_persona_defaults_spoken_replies_to_two_or_three_sentences() -> None:
    from pathlib import Path

    import jarvis.brain as brain_pkg

    text = (Path(brain_pkg.__file__).parent / "JARVIS_PERSONA.md").read_text("utf-8")
    assert "Two to four" not in text
    full, compact = text.split("## System-Prompt (Compact)")
    for variant in (full, compact):
        flat = " ".join(variant.split())
        assert "Two to three short" in flat
        for exception in ("details", "complex", "safety", "question back"):
            assert exception in flat, exception
