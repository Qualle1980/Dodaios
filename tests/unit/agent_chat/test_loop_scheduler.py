"""LoopScheduler: one task per coding agent, one Codex review at a time.

Slice 1 of the coding loop (2026-11-17). Pure logic — no pane I/O.
"""

from __future__ import annotations

from jarvis.agent_chat.loop_scheduler import (
    Acquired,
    LoopScheduler,
    Queued,
)


def test_one_claude_and_one_grok_run_together() -> None:
    sched = LoopScheduler()
    claude = sched.try_acquire("claude", "code")
    grok = sched.try_acquire("grok-build", "code")
    assert isinstance(claude, Acquired)
    assert isinstance(grok, Acquired)
    assert claude.agent == "claude"
    assert grok.agent == "grok-build"
    status = sched.status()
    assert status["slots"]["claude"]["busy"] is True
    assert status["slots"]["grok-build"]["busy"] is True
    assert status["slots"]["codex-review"]["busy"] is False


def test_second_claude_task_is_queued_while_first_is_held() -> None:
    sched = LoopScheduler()
    first = sched.try_acquire("claude", "code")
    second = sched.try_acquire("claude", "code")
    assert isinstance(first, Acquired)
    assert isinstance(second, Queued)
    assert "waiting for the claude slot" in second.message
    assert "(0 ahead)" in second.message or "0 ahead" in second.message
    assert sched.status()["queue_lengths"]["claude"] == 1


def test_codex_review_slot_is_independent_of_coder_slots() -> None:
    sched = LoopScheduler()
    assert isinstance(sched.try_acquire("claude", "code"), Acquired)
    assert isinstance(sched.try_acquire("grok-build", "code"), Acquired)
    review = sched.try_acquire("codex-review", "review")
    assert isinstance(review, Acquired)
    assert review.agent == "codex-review"
    status = sched.status()
    assert status["slots"]["codex-review"]["busy"] is True
    assert status["slots"]["claude"]["busy"] is True
    assert status["slots"]["grok-build"]["busy"] is True


def test_release_starts_the_next_queued_task_for_that_agent() -> None:
    sched = LoopScheduler()
    first = sched.try_acquire("claude", "code")
    second = sched.try_acquire("claude", "code")
    assert isinstance(first, Acquired)
    assert isinstance(second, Queued)
    started = sched.release(first.handle)
    assert isinstance(started, Acquired)
    assert started.agent == "claude"
    assert started.handle == second.handle
    assert sched.status()["queue_lengths"]["claude"] == 0
    assert sched.status()["slots"]["claude"]["busy"] is True
