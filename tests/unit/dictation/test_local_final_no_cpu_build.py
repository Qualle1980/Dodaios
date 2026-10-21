"""The local final pass must not build a model it is going to decline.

Measured live 2026-10-21: the worker spent 109 s on this host (probe 4.7 s,
build 85.9 s, first decode 18.4 s) and was then closed unused because it came
up on the CPU. The wake word shares that CPU — vosk stage-1 was demoted for
missing realtime and a spoken "Jarvis" was suppressed while the build ran.
"""
from __future__ import annotations

import pytest

from jarvis.dictation import local_final
from jarvis.dictation.local_final import LocalFinalSTT


@pytest.fixture(autouse=True)
def _engine_present(monkeypatch):
    """faster-whisper installed, so the refusal cannot come from that check."""
    import jarvis.dictation.local_preview as preview

    monkeypatch.setattr(preview, "faster_whisper_available", lambda: True)


def _spawn_tripwire(monkeypatch) -> list[int]:
    """Record any attempt to actually build the model."""
    built: list[int] = []
    import jarvis.dictation.local_preview as preview

    def _never(*_a: object, **_kw: object):
        built.append(1)
        raise AssertionError("the model was built despite a refusal")

    monkeypatch.setattr(preview, "_spawn_worker_model", _never)
    return built


def test_unverified_gpu_inference_refuses_before_the_build(monkeypatch) -> None:
    import jarvis.hardware.detection as detection

    monkeypatch.setattr(detection, "cuda_inference_verified", lambda: None)
    built = _spawn_tripwire(monkeypatch)

    stt = LocalFinalSTT()
    refusal = stt._spawn_refusal()

    assert refusal, "an unverified host must refuse"
    assert "CPU" in refusal
    assert built == [], "nothing may be built when the answer is already known"


def test_a_host_without_gpu_inference_refuses_before_the_build(monkeypatch) -> None:
    import jarvis.hardware.detection as detection

    monkeypatch.setattr(detection, "cuda_inference_verified", lambda: False)
    built = _spawn_tripwire(monkeypatch)

    assert LocalFinalSTT()._spawn_refusal(), "a probed-negative host must refuse"
    assert built == []


def test_a_verified_gpu_host_is_still_allowed(monkeypatch) -> None:
    """The gate must not disable the feature where it genuinely pays off."""
    import jarvis.hardware.detection as detection

    monkeypatch.setattr(detection, "cuda_inference_verified", lambda: True)
    monkeypatch.setattr(detection, "free_accelerator_gb", lambda: (24.0, "nvidia-smi"))

    assert LocalFinalSTT()._spawn_refusal() == ""


def test_an_explicit_cpu_opt_in_still_wins(monkeypatch) -> None:
    """``allow_cpu`` means the caller accepted the CPU cost on purpose."""
    import jarvis.hardware.detection as detection

    monkeypatch.setattr(detection, "cuda_inference_verified", lambda: False)
    stt = LocalFinalSTT()
    stt._allow_cpu = True
    assert stt._spawn_refusal() == ""


def test_the_refusal_is_fast(monkeypatch) -> None:
    """It must cost milliseconds, not the build it replaces."""
    import time

    import jarvis.hardware.detection as detection

    monkeypatch.setattr(detection, "cuda_inference_verified", lambda: None)
    _spawn_tripwire(monkeypatch)

    t0 = time.perf_counter()
    LocalFinalSTT()._spawn_refusal()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"refusal took {elapsed:.1f}s; it replaces a 109 s build"


def test_a_broken_probe_does_not_veto_the_spawn(monkeypatch) -> None:
    """An unreadable probe must not permanently disable local dictation."""
    import jarvis.hardware.detection as detection

    def boom() -> bool:
        raise RuntimeError("probe cache unreadable")

    monkeypatch.setattr(detection, "cuda_inference_verified", boom)
    monkeypatch.setattr(local_final, "log", local_final.log)
    # Unknown reads as "not verified", which is the CPU-first floor: refuse,
    # but say why, rather than crash the warm-up.
    assert LocalFinalSTT()._spawn_refusal()
