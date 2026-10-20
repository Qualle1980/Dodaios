"""A blocked write must never pass as finished coding work.

Captured live on 2026-10-20: ``codex exec`` was told to create a file, its
sandbox refused the write, and the process still exited 0 with a
``turn.completed`` event. Process exit is therefore not evidence that coding
work happened, and the stream below is the real stdout of that run.

What the existing chain does with it, each step asserted here:

* the blocked attempt produces NO action item at all — only ``agent_message``
  text — so there is no tool-call evidence to defer an empty diff to the LLM
  critic with, and the runner's deterministic empty-diff revise stands;
* an action item that IS present but reports failure is removed from the
  evidence set, so a denied attempt cannot stand in for a completed one;
* the request shape decides the read-only carve-out, never the worker's claim:
  a "create a file" task is not informational, so an empty diff keeps its
  deterministic revise, while a question is informational and may be answered
  with no diff at all.

No code was changed for this — the mechanisms were already in place. These
tests pin them against the captured stream so the case can never regress
without being caught, and never needs another paid run to check.

Deterministic — no network, no provider, no model.
"""
from __future__ import annotations

from jarvis.missions.critic.runner import _extract_tool_call_evidence
from jarvis.missions.stream_evidence import is_informational_request

_WRITE_TASK = (
    "Create a file named hello.txt in the current directory containing exactly: hi"
)

#: Verbatim stdout of the blocked run (see the module docstring).
BLOCKED_WRITE_STREAM = "\n".join(
    [
        '{"type":"thread.started","thread_id":"01a158f0-10b6-70a1-881d-b3ac41954056"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
        '"text":"I will try to create hello.txt with exactly hi."}}',
        '{"type":"item.completed","item":{"id":"item_1","type":"agent_message",'
        '"text":"Could not create hello.txt: the read-only workspace policy '
        'blocked the write."}}',
        '{"type":"turn.completed","usage":{"input_tokens":36930,"output_tokens":97}}',
    ]
)

#: The same task, succeeding: codex reports a file_change action item.
SUCCESSFUL_WRITE_STREAM = "\n".join(
    [
        '{"type":"thread.started","thread_id":"t1"}',
        '{"type":"item.completed","item":{"id":"item_1","type":"file_change",'
        '"status":"completed","changes":[{"path":"hello.txt"}]}}',
        '{"type":"turn.completed"}',
    ]
)

#: An action item that ran and failed (non-zero exit).
FAILED_COMMAND_STREAM = "\n".join(
    [
        '{"type":"item.completed","item":{"id":"item_1",'
        '"type":"command_execution","status":"failed","exit_code":1,'
        '"command":"powershell -Command ..."}}',
        '{"type":"turn.completed"}',
    ]
)


def test_the_captured_stream_really_is_a_clean_exit_with_no_work() -> None:
    """The premise of this file, pinned: the turn completed and did nothing."""
    assert '"type":"turn.completed"' in BLOCKED_WRITE_STREAM
    assert '"type":"file_change"' not in BLOCKED_WRITE_STREAM
    assert '"type":"command_execution"' not in BLOCKED_WRITE_STREAM


def test_a_blocked_write_leaves_no_tool_call_evidence() -> None:
    """Without evidence the empty diff is not deferred to the LLM critic."""
    assert _extract_tool_call_evidence(BLOCKED_WRITE_STREAM) == ()


def test_a_successful_write_does_leave_evidence() -> None:
    """Control: the assertion above is about the blocked run, not the parser."""
    assert _extract_tool_call_evidence(SUCCESSFUL_WRITE_STREAM) == ("file_change",)


def test_a_failed_action_item_is_removed_from_the_evidence() -> None:
    """An attempted action is not proof that the action completed."""
    assert _extract_tool_call_evidence(FAILED_COMMAND_STREAM) == ()


def test_a_write_task_is_not_informational_so_an_empty_diff_still_vetoes() -> None:
    """The read-only carve-out keys off the REQUEST, never the worker's claim."""
    assert is_informational_request(_WRITE_TASK) is False


def test_a_question_is_informational_and_needs_no_diff() -> None:
    """Analysis and read-only work stays able to succeed without a diff."""
    assert (
        is_informational_request(
            "which city would you recommend for a trip to Australia?"
        )
        is True
    )
