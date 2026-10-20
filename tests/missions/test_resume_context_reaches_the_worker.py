"""A resumed step keeps the context the first attempt earned (phase 7).

Three things have to survive a stop — a capacity park or a crash — and reach
the worker that continues:

* the **critic's prior feedback**, which is file-backed in the mission
  directory, so a fresh ``ReflectionMemory`` over the same directory still
  renders it after a restart;
* the **note that this is a continuation**, so the worker inspects the
  workspace instead of starting over;
* and nothing else: the note must not claim a CAUSE, because since the event
  log can replay a checkpoint after a crash, "the provider ran out of
  capacity" would be untrue for half the resumes.

Deterministic — no worker, no model, no network.
"""
from __future__ import annotations

from pathlib import Path

from jarvis.missions.critic.reflections import ReflectionMemory
from jarvis.missions.kontrollierer.orchestrator import _RESUME_NOTE
from jarvis.missions.kontrollierer.worker_prompt import compose_worker_prompt


def _write_reflection(mission_dir: Path) -> None:
    memory = ReflectionMemory(mission_dir)
    memory.append(
        iteration=0,
        summary="the endpoint has no test",
        evidence=["add a test for the new route"],
    )


def test_prior_feedback_survives_a_restart(tmp_path: Path) -> None:
    """A new process reads the same file and sees the earlier iteration."""
    _write_reflection(tmp_path)

    after_restart = ReflectionMemory(tmp_path).render_for_worker_prompt(n=3)

    assert "add a test for the new route" in after_restart


def test_the_resumed_worker_is_told_both_the_note_and_the_feedback(
    tmp_path: Path,
) -> None:
    _write_reflection(tmp_path)
    prior = ReflectionMemory(tmp_path).render_for_worker_prompt(n=3)
    prior_block = "\n\n".join(block for block in (_RESUME_NOTE, prior) if block)

    prompt = compose_worker_prompt(prior_block, "add a health endpoint")

    assert "RESUMED TASK" in prompt
    assert "add a test for the new route" in prompt
    assert "add a health endpoint" in prompt
    assert prompt.index("RESUMED TASK") < prompt.index("add a health endpoint")


def test_the_resume_note_states_no_cause() -> None:
    """A crash resume and a capacity resume read the same note, so it may not
    name a cause that is only true for one of them."""
    assert "capacity" not in _RESUME_NOTE.lower()
    assert "stopped before it was finished" in _RESUME_NOTE


def test_a_first_attempt_gets_no_resume_framing(tmp_path: Path) -> None:
    """An empty memory renders nothing — no boilerplate, no wasted tokens."""
    assert ReflectionMemory(tmp_path).render_for_worker_prompt(n=3) == ""

    prompt = compose_worker_prompt("", "add a health endpoint")

    assert "RESUMED TASK" not in prompt
