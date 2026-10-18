"""On-device text-to-speech: Supertonic 3 (INT8) through sherpa-onnx.

A second keyless local voice next to Piper. Where a Piper voice speaks one
language, ONE Supertonic bundle speaks 31 (German, English and Arabic among
them) with ten built-in speakers: sids 0-4 are feminine (F1-F5), 5-9 masculine
(M1-M5). The language is passed per call, so the turn's output language from
``resolve_output_language`` maps straight onto the synthesiser.

sherpa-onnx's Supertonic path fires its progress callback once, at the end, so
this adapter streams by sentence: each sentence is synthesised and yielded
before the next one starts, which keeps time-to-first-audio at one sentence.

Licensing: the model weights are OpenRAIL-M (use restrictions, free of
charge); the bundle ships its LICENSE file next to the weights.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from jarvis.core.protocols import AudioChunk
from jarvis.speech.local_models import SUPERTONIC_MODEL_ID

log = logging.getLogger(__name__)

_INT16_FULL_SCALE = 32767.0
_DEFAULT_SPEED = 1.0

#: Built-in speaker ids per voice profile.
_SPEAKERS = {
    "feminine": {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4},
    "masculine": {"M1": 5, "M2": 6, "M3": 7, "M4": 8, "M5": 9},
}
_DEFAULT_SPEAKER = {"feminine": "F1", "masculine": "M1"}

#: Languages the Supertonic 3 bundle speaks (ISO 639-1).
SUPPORTED_LANGUAGES = frozenset(
    {
        "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
        "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    }
)

_SENTENCE_RE = re.compile(r"(?<=[.!?؟۔。])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split *text* into speakable sentences (Latin, Arabic and CJK stops)."""
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


#: Every built-in speaker name, for config allowlists.
SPEAKER_NAMES = frozenset(name for group in _SPEAKERS.values() for name in group)


def speaker_id(voice: str | None, gender: str) -> tuple[str, int]:
    """Resolve a voice name (``M1``..``F5``) or a profile to a speaker id."""
    name = (voice or "").strip().upper()
    for speakers in _SPEAKERS.values():
        if name in speakers:
            return name, speakers[name]
    profile = gender if gender in _SPEAKERS else "masculine"
    name = _DEFAULT_SPEAKER[profile]
    return name, _SPEAKERS[profile][name]


class SupertonicLocalTTS:
    """Local multilingual text-to-speech (keyless, on-device, ten speakers)."""

    name = "supertonic-local"
    supports_streaming = True
    last_voice: str | None = None
    last_voice_provider: str | None = None
    runs_on_device = True

    def __init__(
        self,
        *,
        voice: str | None = None,
        gender: str | None = None,
        num_threads: int = 2,
        speed: float = _DEFAULT_SPEED,
    ) -> None:
        self._preferred_voice = (voice or "").strip() or None
        self._gender = (gender or "").strip().lower() or "masculine"
        self._num_threads = max(1, int(num_threads))
        self._speed = float(speed) if speed and speed > 0 else _DEFAULT_SPEED
        self._engine: Any = None
        self._load_lock = asyncio.Lock()
        # AP-24: one native engine per instance, never driven by two callers.
        self._synth_lock = threading.Lock()

    def list_voices(self, language: str | None = None) -> list[str]:
        """Built-in speakers; empty when the bundle or language is unavailable."""
        from jarvis.speech.local_models import bundle_present

        if not bundle_present(SUPERTONIC_MODEL_ID):
            return []
        if language and language.lower().split("-", 1)[0] not in SUPPORTED_LANGUAGES:
            return []
        return [name for speakers in _SPEAKERS.values() for name in speakers]

    # -- engine -------------------------------------------------------------
    def _build_engine(self) -> Any:
        try:
            import sherpa_onnx  # noqa: PLC0415 — optional engine, lazy on purpose
        except ImportError as exc:
            raise RuntimeError(
                "The local speech runtime (sherpa-onnx) is not installed, so the "
                "on-device voice cannot speak. Install it from the API-Keys view."
            ) from exc

        from jarvis.speech.sherpa_models import model_paths

        paths = model_paths(SUPERTONIC_MODEL_ID)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise RuntimeError(
                "The Supertonic voice is not fully downloaded (missing: "
                f"{', '.join(sorted(missing))}). Download it from the API-Keys view."
            )
        log.info("Loading local Supertonic voice %s", SUPERTONIC_MODEL_ID)
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=str(paths["duration_predictor.int8.onnx"]),
                    text_encoder=str(paths["text_encoder.int8.onnx"]),
                    vector_estimator=str(paths["vector_estimator.int8.onnx"]),
                    vocoder=str(paths["vocoder.int8.onnx"]),
                    tts_json=str(paths["tts.json"]),
                    unicode_indexer=str(paths["unicode_indexer.bin"]),
                    voice_style=str(paths["voice.bin"]),
                ),
                num_threads=self._num_threads,
                provider="cpu",
            ),
        )
        return sherpa_onnx.OfflineTts(config)

    async def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        async with self._load_lock:
            if self._engine is None:
                self._engine = await asyncio.to_thread(self._build_engine)
        return self._engine

    def _generate(self, engine: Any, sentence: str, sid: int, language: str) -> Any:
        import sherpa_onnx  # noqa: PLC0415 — engine already proved it is installed

        config = sherpa_onnx.GenerationConfig()
        config.sid = sid
        config.speed = self._speed
        config.extra = {"lang": language}
        if not self._synth_lock.acquire(blocking=False):
            raise RuntimeError("The Supertonic voice is busy with another turn.")
        try:
            return engine.generate(sentence, config)
        finally:
            self._synth_lock.release()

    # -- synthesis ----------------------------------------------------------
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language_code: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Speak *text* sentence by sentence in this turn's output language."""
        sentences = split_sentences(text or "")
        if not sentences:
            return
        language = (language_code or "").strip().lower().split("-", 1)[0] or "en"
        if language not in SUPPORTED_LANGUAGES:
            log.warning(
                "Supertonic does not speak %r; using English for this turn.",
                language,
            )
            language = "en"
        name, sid = speaker_id(voice or self._preferred_voice, self._gender)
        engine = await self._get_engine()
        self.last_voice = name
        self.last_voice_provider = self.name
        for sentence in sentences:
            audio = await asyncio.to_thread(
                self._generate, engine, sentence, sid, language
            )
            samples = np.asarray(getattr(audio, "samples", ()), dtype=np.float32)
            rate = int(getattr(audio, "sample_rate", 0) or 0)
            if samples.size == 0:
                continue
            if rate <= 0:
                raise RuntimeError("The local voice returned no sample rate.")
            pcm = (np.clip(samples, -1.0, 1.0) * _INT16_FULL_SCALE).astype(np.int16)
            yield AudioChunk(
                pcm=pcm.tobytes(), sample_rate=rate, timestamp_ns=0, channels=1
            )

    async def _ensure_client(self) -> None:
        """Warm-up hook; must never break boot when the bundle is missing."""
        from jarvis.speech.local_models import bundle_present

        if not bundle_present(SUPERTONIC_MODEL_ID):
            return
        try:
            await self._get_engine()
        except Exception as exc:  # noqa: BLE001 — warm-up must never break boot
            log.debug("Supertonic warm-up skipped (%s).", exc)

    async def aclose(self) -> None:
        """Drop the loaded engine so a switched-away provider frees its memory."""
        self._engine = None
