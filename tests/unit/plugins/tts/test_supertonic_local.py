"""Supertonic local TTS: sentence streaming, speaker and language selection."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from jarvis.plugins.tts.supertonic_local import (
    SupertonicLocalTTS,
    speaker_id,
    split_sentences,
)


@dataclass
class _Audio:
    samples: list[float]
    sample_rate: int = 44100


class _RecordingTTS(SupertonicLocalTTS):
    """Skips the native engine; records each sentence it is asked to speak."""

    def __init__(self) -> None:
        super().__init__(gender="feminine")
        self._engine = object()
        self.calls: list[tuple[str, int, str]] = []

    def _generate(self, engine, sentence, sid, language):  # type: ignore[override]
        self.calls.append((sentence, sid, language))
        return _Audio(samples=[0.0, 0.5, -0.5])


def _collect(tts: SupertonicLocalTTS, text: str, **kwargs) -> list:
    async def run() -> list:
        return [chunk async for chunk in tts.synthesize(text, **kwargs)]

    return asyncio.run(run())


def test_split_sentences_handles_latin_and_arabic_stops() -> None:
    assert split_sentences("Hallo. Wie geht's? Gut!") == ["Hallo.", "Wie geht's?", "Gut!"]
    assert split_sentences("مرحبا؟ كيف حالك") == ["مرحبا؟", "كيف حالك"]
    assert split_sentences("  \n ") == []


def test_speaker_id_prefers_pinned_name_then_profile() -> None:
    assert speaker_id("m3", "feminine") == ("M3", 7)
    assert speaker_id(None, "feminine") == ("F1", 0)
    assert speaker_id("Charon", "unknown") == ("M1", 5)


def test_synthesize_yields_one_chunk_per_sentence_in_turn_language() -> None:
    tts = _RecordingTTS()
    chunks = _collect(tts, "Erster Satz. Zweiter Satz.", language_code="de-DE")
    assert [c[0] for c in tts.calls] == ["Erster Satz.", "Zweiter Satz."]
    assert {c[1:] for c in tts.calls} == {(0, "de")}
    assert len(chunks) == 2
    assert chunks[0].sample_rate == 44100
    assert len(chunks[0].pcm) == 6  # three int16 samples
    assert tts.last_voice == "F1"
    assert tts.last_voice_provider == "supertonic-local"


def test_unsupported_language_falls_back_to_english() -> None:
    tts = _RecordingTTS()
    _collect(tts, "Hola.", language_code="xx")
    assert tts.calls[0][2] == "en"


def test_arabic_is_passed_through() -> None:
    tts = _RecordingTTS()
    _collect(tts, "مرحبا.", language_code="ar")
    assert tts.calls[0][2] == "ar"
