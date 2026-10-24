"""Codex ``exec --json`` items read by the upstream status semantics.

Live 2026-10-23: ``rg -n TODO budget.py`` found nothing, exited 1, and Codex
reported the item as ``status: "failed"``. The critic treated that as "never
ran" and logged "no tool calls" for a worker that had searched the file.
A failed command still proves no side effect (test_capability_honesty); only
the read-order gate counts a lookup that ran.
"""
from __future__ import annotations

import json

from jarvis.missions.critic.runner import _read_evidence_ran, _read_task_verdict
from jarvis.missions.workers.codex_direct_worker import _sandbox_for
from jarvis.plugins.tool.spawn_worker import _build_mission_prompt

_READ_PROMPT = _build_mission_prompt(
    "Lass Codex die TODO-Kommentare in jarvis/missions/budget.py auflisten. "  # i18n-allow
    "Nur lesen, nichts ändern.",  # i18n-allow
    action="",
)


def _codex_stream(*, status: str, exit_code: int | None, answer: str) -> str:
    command = {
        "id": "item_1", "type": "command_execution",
        "command": "rg -n TODO jarvis/missions/budget.py",
        "aggregated_output": "", "exit_code": exit_code, "status": status,
    }
    frames = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.started", "item": {**command, "status": "in_progress", "exit_code": None}},
        {"type": "item.completed", "item": command},
        {"type": "item.completed",
         "item": {"id": "item_2", "type": "agent_message", "text": answer}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    return "\n".join(json.dumps(f) for f in frames)


def test_a_search_without_matches_still_counts_as_run() -> None:
    stream = _codex_stream(status="failed", exit_code=1, answer="0 TODOs found.")
    assert _read_evidence_ran(stream)


def test_a_declined_command_never_ran() -> None:
    stream = _codex_stream(status="declined", exit_code=None, answer="0 TODOs found.")
    assert not _read_evidence_ran(stream)


def test_a_read_answer_backed_by_a_search_passes_this_gate() -> None:
    stream = _codex_stream(status="failed", exit_code=1, answer="0 TODOs found in budget.py.")
    assert _read_task_verdict(_READ_PROMPT, "", stream) is None


def test_a_read_answer_without_any_search_is_revised() -> None:
    frames = [{"type": "item.completed", "item": {
        "id": "item_0", "type": "agent_message", "text": "0 TODOs found."}}]
    verdict = _read_task_verdict(_READ_PROMPT, "", json.dumps(frames[0]))

    assert verdict is not None
    assert verdict.verdict == "revise"
    assert "read or search" in verdict.correction_instruction


def test_a_read_order_runs_in_the_read_only_sandbox() -> None:
    assert _sandbox_for(_READ_PROMPT) == "read-only"
    assert _sandbox_for(_build_mission_prompt("Fix the typo in README.md", action="")) == (
        "workspace-write"
    )


def test_the_sandbox_follows_the_request_inside_the_real_worker_prompt() -> None:
    """Live 2026-10-24: the framing directives read as read-only, so the
    composed prompt put EVERY order in the read-only sandbox and a plain
    "create docs/tmp_probe.md" failed three times."""
    from jarvis.missions.kontrollierer.worker_prompt import compose_worker_prompt

    feedback = "Do not modify unrelated files. The earlier attempt wrote nothing."
    change = "Create the file docs/tmp_probe.md containing exactly the line 'probe'."
    for prior in ("", feedback):
        assert _sandbox_for(compose_worker_prompt(prior, change)) == "workspace-write"
        assert _sandbox_for(compose_worker_prompt(prior, _READ_PROMPT)) == "read-only"
        assert _sandbox_for(compose_worker_prompt(
            prior, "List the TODOs in jarvis/missions/budget.py")) == "read-only"
