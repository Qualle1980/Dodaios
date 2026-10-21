"""A finished turn must not swallow the next question.

Measured live 2026-10-21. After every answer the microphone picked up the
assistant's own speech tail and the VAD reported "speech start" 0.4-1.3 s
later at rms 0.003-0.034 (near-silence, not the user). That freeze cleared the
continuation deadline outright, which armed the window forever, so the next
real question — six seconds later — was glued onto the previous one:

    transcript final: 'Wie viel ist 10 plus 10?'
    Continuation recombine -> 'wie viel ist 100 plus 100? كم هو 2 مقارنة مع 2؟
                               Wie viel ist 10 plus 10?'
    Jarvis: 'Hundert plus hundert ist zweihundert. Zwei ist gleich zwei.
             Zehn plus zehn ist zwanzig.'

Three questions answered at once, and the local zero-token answers bypassed
because the combined text matches no one-shot pattern.
"""
from __future__ import annotations

import pytest

from jarvis.speech.continuation_window import (
    ContinuationWindow,
    is_self_contained_request,
)


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def advance_ms(self, ms: float) -> None:
        self.ns += int(ms * 1_000_000)


def _armed(clk: FakeClock, prior: str = "wie viel ist 100 plus 100?"):
    w = ContinuationWindow(grace_ms=2500, max_chain=8, clock=clk)
    w.note_dispatch(prior, continued=False)
    return w


# ---- the exact live failures ----------------------------------------------

@pytest.mark.parametrize(
    ("label", "to_idle_ms", "to_vad_ms", "to_final_ms", "text"),
    [
        ("3.00 s gap", 404, 412, 2590, "Wie spät ist das?"),
        ("6.01 s gap", 4081, 960, 5046, "كم هو 2 مقارنة مع 2؟"),
        ("6.65 s gap", 229, 1323, 5326, "Wie viel ist 10 plus 10?"),
    ],
)
def test_the_live_false_merges_no_longer_happen(
    label: str, to_idle_ms: int, to_vad_ms: int, to_final_ms: int, text: str
) -> None:
    clk = FakeClock()
    w = _armed(clk)
    clk.advance_ms(to_idle_ms)
    w.mark_idle()
    clk.advance_ms(to_vad_ms)
    w.note_speech_resumed()  # the assistant's own speech tail
    clk.advance_ms(to_final_ms)
    assert w.try_recombine(text) is None, label


def test_a_spurious_freeze_cannot_arm_the_window_forever() -> None:
    """Repeated noise used to hold the window open for the whole session."""
    clk = FakeClock()
    w = _armed(clk)
    w.mark_idle()
    for _ in range(10):
        clk.advance_ms(900)
        w.note_speech_resumed()
    assert w.is_live() is False
    assert w.try_recombine("Wie spät ist es?") is None


def test_the_freeze_still_buys_time_to_formulate() -> None:
    """The bug this freeze fixed (session 71f2d2de) must stay fixed: ~3 s to
    formulate the follow-up is longer than the 2.5 s grace."""
    clk = FakeClock()
    w = _armed(clk, "erzähl mir vom weitesten Ort")
    w.mark_idle()
    clk.advance_ms(1000)
    w.note_speech_resumed()
    clk.advance_ms(3000)  # 4 s total, past the plain grace
    assert w.try_recombine("nicht australische") == (
        "erzähl mir vom weitesten Ort nicht australische"
    )


# ---- the semantic guard ----------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Wie viel ist 10 plus 10?",
        "Wie spät ist das?",
        "Was ist 10 plus 10?",
        "Kannst du mal sagen, wer ich bin?",
        "Erzähl mir etwas Interessantes.",
        "Welcher Tag ist heute?",
        "Öffne Notepad",
        "What time is it?",
        "Can you tell me who I am?",
        "كم هو 2 مقارنة مع 2؟",
        "Danke.",
    ],
)
def test_a_complete_request_is_never_a_continuation(text: str) -> None:
    assert is_self_contained_request(text) is True
    clk = FakeClock()
    w = _armed(clk)
    w.mark_idle()
    clk.advance_ms(200)  # well inside the grace
    assert w.try_recombine(text) is None, "a whole new request must stand alone"


@pytest.mark.parametrize(
    "text",
    [
        "nicht australische",
        "sondern der weiteste Ort",
        "und zwar schnell",
        "australien oder so, nein",
        "also eher der zweite",
        "mit dem blauen Deckel",
    ],
)
def test_a_fragment_is_still_treated_as_a_continuation(text: str) -> None:
    """The feature's own examples must keep working."""
    assert is_self_contained_request(text) is False
    clk = FakeClock()
    w = _armed(clk, "erzähl mir vom weitesten Ort")
    w.mark_idle()
    clk.advance_ms(200)
    assert w.try_recombine(text) is not None


def test_empty_text_is_not_a_request() -> None:
    assert is_self_contained_request("") is False
    assert is_self_contained_request("   ") is False


def test_the_guard_survives_punctuation_and_quoting() -> None:
    assert is_self_contained_request('"Wie spät ist es?"') is True
    assert is_self_contained_request("- was ist das") is True
