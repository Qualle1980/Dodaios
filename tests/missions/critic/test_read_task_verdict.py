"""A read order is judged on its answer, not on an empty diff.

Live 2026-10-23: "list the TODOs in budget.py, read only" was answered with
the whole file (it has no TODO) and the critic approved it at 0.95.
"""
from __future__ import annotations

import json

from jarvis.missions.critic.prompts import READ_TASK_RULE, render_critic_prompt
from jarvis.missions.critic.runner import _read_task_verdict
from jarvis.plugins.tool.spawn_worker import _build_mission_prompt

_READ_PROMPT = _build_mission_prompt(
    "Lass Codex die TODO-Kommentare in jarvis/missions/budget.py auflisten. "  # i18n-allow
    "Nur lesen, nichts ändern.",  # i18n-allow
    action="",
)
_CHANGE_PROMPT = _build_mission_prompt("Fix the typo in README.md", action="")


def _stream(answer: str) -> str:
    """A worker stream with one Read tool call and a final answer."""
    frames = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Read",
             "input": {"file_path": "jarvis/missions/budget.py"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "..."},
        ]}},
        {"type": "result", "result": answer},
    ]
    return "\n".join(json.dumps(f) for f in frames)


_DUMP = "Gefunden: jarvis/missions/budget.py\n\n```python\n" + "\n".join(  # i18n-allow
    f"line_{i} = {i}" for i in range(40)
) + "\n```"


def test_a_file_dump_is_revised_without_an_llm() -> None:
    verdict = _read_task_verdict(_READ_PROMPT, "", _stream(_DUMP))

    assert verdict is not None
    assert verdict.verdict == "revise"
    assert verdict.axes["correctness"].status == "fail"
    assert "0 TODOs found" in verdict.correction_instruction


def test_a_change_during_a_read_order_is_revised() -> None:
    diff = (
        "diff --git a/jarvis/missions/budget.py b/jarvis/missions/budget.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    verdict = _read_task_verdict(_READ_PROMPT, diff, _stream("0 TODOs found."))

    assert verdict is not None
    assert verdict.axes["side_effects"].status == "fail"


def test_a_plain_finding_goes_on_to_the_critic() -> None:
    answer = "0 TODOs found in jarvis/missions/budget.py (searched for TODO, FIXME)."
    assert _read_task_verdict(_READ_PROMPT, "", _stream(answer)) is None


def test_a_change_order_is_not_judged_here() -> None:
    assert _read_task_verdict(_CHANGE_PROMPT, "", _stream(_DUMP)) is None


def _render(mission_prompt: str) -> str:
    return render_critic_prompt(
        mission_prompt=mission_prompt, worker_diff="", log_summary="",
        prior_reflections="", iteration=0,
    )


def test_the_critic_gets_the_read_rule_and_the_users_own_words() -> None:
    prompt = _render(_READ_PROMPT)

    assert "READ-TASK RULE" in prompt
    assert "An empty diff is expected and is NOT evidence of quality" in prompt
    assert "USER'S OWN REQUEST: <<<Lass Codex" in prompt  # i18n-allow


def test_a_change_order_gets_no_read_rule() -> None:
    assert READ_TASK_RULE.strip()[:20] not in _render(_CHANGE_PROMPT)
