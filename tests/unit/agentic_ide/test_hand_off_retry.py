"""RED: one model retry after refusal, then stop; never same id twice."""

from __future__ import annotations

from jarvis.agentic_ide.hand_off_retry import (
    MAX_OPENS_PER_REVIEW,
    HandOffRetryGate,
    review_key,
)


def test_refusal_allows_exactly_one_retry_with_different_model() -> None:
    gate = HandOffRetryGate()
    key = review_key(owner_session="s1", fingerprint="fp1")
    assert gate.allow_open(key, "gpt-6.1-sol") is None
    gate.note_open(key, "gpt-6.1-sol")
    gate.note_refusal(key, "gpt-6.1-sol", reason="could not be confirmed")
    # Same id must not be retried.
    same = gate.allow_open(key, "gpt-6.1-sol")
    assert same is not None
    assert "review not run" in same.lower()
    # Different confirmed id: one retry allowed.
    assert gate.allow_open(key, "gpt-5.6-luna") is None
    gate.note_open(key, "gpt-5.6-luna")
    gate.note_refusal(key, "gpt-5.6-luna", reason="could not be confirmed")
    stopped = gate.allow_open(key, "gpt-6-sol")
    assert stopped is not None
    assert "review not run" in stopped.lower()


def test_second_refusal_ends_with_review_not_run() -> None:
    gate = HandOffRetryGate()
    key = review_key(owner_session="s1", title="hallo3")
    gate.note_refusal(key, "a")
    gate.note_refusal(key, "b")
    msg = gate.allow_open(key, "c")
    assert msg is not None
    assert msg.startswith("review not run:")


def test_opened_panes_at_most_two_per_review() -> None:
    gate = HandOffRetryGate()
    key = review_key(owner_session="owner", fingerprint="f")
    assert MAX_OPENS_PER_REVIEW == 2
    assert gate.allow_open(key, "m1") is None
    gate.note_open(key, "m1")
    assert gate.allow_open(key, "m2") is None
    gate.note_open(key, "m2")
    blocked = gate.allow_open(key, "m3")
    assert blocked is not None
    assert "review not run" in blocked.lower()
