"""What may count as acceptance of delegated coding work, and what may not."""

from __future__ import annotations

from uuid import uuid4

import pytest

from jarvis.agent_chat.delegation_acceptance import (
    AcceptanceGate,
    Evidence,
    EvidenceWatch,
    claims_acceptance,
    gate_for,
    reads_version_control,
    runs_tests,
    unverified_answer,
)
from jarvis.core.bus import EventBus
from jarvis.core.events import ActionExecuted, ActionProposed

COMPLETED_ROW = {
    "task_id": "coding:abc:1:2",
    "status": "completed",
    "report": "Added subtract().",
    "evidence": "Recorded CLI completion and matching report; not independent verification",
}


# ------------------------------------------------------------- when it arms


def test_an_unverified_completion_arms_the_gate() -> None:
    gate = gate_for([COMPLETED_ROW])
    assert gate is not None
    assert gate.work == ("coding:abc:1:2",)


@pytest.mark.parametrize(
    "row",
    [
        {"task_id": "t", "status": "needs_input", "report": "May I edit calc.py?"},
        {"task_id": "t", "status": "superseded", "report": "The pane restarted."},
        {"task_id": "t", "status": "error", "report": "crashed"},
        {"task_id": "t", "status": "timed_out", "report": "no completion"},
        {"task_id": "t", "status": "unverified", "report": "no local record"},
    ],
)
def test_a_row_that_claims_nothing_needs_no_gate(row) -> None:
    """Only a completion carries an acceptance to guard."""
    assert gate_for([row]) is None


def test_a_completion_that_was_independently_verified_needs_no_gate() -> None:
    """Should a handback ever carry real evidence, it is not second-guessed."""
    row = {**COMPLETED_ROW, "evidence": "Parent ran the suite on the same commit"}
    assert gate_for([row]) is None


def test_one_unverified_row_among_many_still_arms() -> None:
    rows = [{"task_id": "x", "status": "needs_input", "report": "?"}, COMPLETED_ROW]
    gate = gate_for(rows)
    assert gate is not None and gate.work == ("coding:abc:1:2",)


# ------------------------------------------------------------ what is a claim


@pytest.mark.parametrize(
    "text",
    [
        "PASS — subtract() is implemented and all tests pass.",
        "All tests pass, we are good.",
        "Verified: the change is in.",
        "Erledigt, alles grün.",  # i18n-allow: German chat output
        "Die Tests laufen durch.",  # i18n-allow: German chat output
        "Geprüft und übernommen.",  # i18n-allow: German chat output
        "Todas las pruebas pasan.",  # i18n-allow: Spanish chat output
    ],
)
def test_a_verdict_is_a_claim(text) -> None:
    assert claims_acceptance(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "T1 reports that it added subtract() to calc.py.",
        "The agent says it is done; I have not verified that.",
        "It stopped at a permission prompt, so this is unverified.",
        "Der Agent meldet sich fertig, ist aber nicht geprüft.",  # i18n-allow: DE output
        "Blocked: the pane needs a decision from you.",
    ],
)
def test_a_report_or_a_disclaimer_is_not_a_claim(text) -> None:
    """A descriptive handback and an honest limit both pass through untouched."""
    assert not claims_acceptance(text)


# ---------------------------------------------------------- what is evidence


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git status --porcelain",
        "git show HEAD",
        "git -C /tmp/p log --oneline -1",
        "git rev-parse HEAD",
    ],
)
def test_a_reading_git_command_is_version_control_evidence(command) -> None:
    assert reads_version_control("RunCommand", {"command": command})


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m x",
        "git checkout main",
        "git push",
        "ls -la",
        "cat calc.py",
        "echo git diff is what I would run",
    ],
)
def test_a_write_or_an_unrelated_command_is_not(command) -> None:
    """Writing is not looking, and talking about git is not running it."""
    assert not reads_version_control("RunCommand", {"command": command})


def test_the_git_tool_itself_counts() -> None:
    assert reads_version_control("workspace-git", {"action": "changes"})


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q",
        "pytest tests/unit",
        "npm test",
        "npx vitest run",
        "cargo test",
        "go test ./...",
    ],
)
def test_a_test_runner_is_test_evidence(command) -> None:
    assert runs_tests("RunCommand", {"command": command})


@pytest.mark.parametrize("command", ["python -c 'import calc'", "git diff", "ruff check ."])
def test_an_import_check_or_a_linter_is_not_a_test_run(command) -> None:
    assert not runs_tests("RunCommand", {"command": command})


# ---------------------------------------------------------------- the verdict


def _evidence(*, vcs: bool, tests: bool) -> Evidence:
    return Evidence(version_control=vcs, tests=tests)


@pytest.mark.parametrize(
    ("vcs", "tests"), [(False, False), (True, False), (False, True)]
)
def test_half_the_evidence_does_not_carry_a_pass(vcs, tests) -> None:
    """A diff without tests proves something was written, not that it works."""
    gate = AcceptanceGate(work=("coding:abc",))
    verdict = gate.verdict("PASS, all tests pass.", _evidence(vcs=vcs, tests=tests))
    assert verdict is not None
    assert not claims_acceptance(verdict)


def test_both_facts_let_the_answer_stand() -> None:
    gate = AcceptanceGate(work=("coding:abc",))
    assert gate.verdict("PASS, verified.", _evidence(vcs=True, tests=True)) is None


def test_an_honest_answer_is_never_rewritten() -> None:
    """Even with no evidence: it claimed nothing, so there is nothing to correct."""
    gate = AcceptanceGate(work=("coding:abc",))
    text = "T1 says it is done; I have not checked the diff yet."
    assert gate.verdict(text, _evidence(vcs=False, tests=False)) is None


@pytest.mark.parametrize("language", ["en", "de", "es", "pt"])
def test_the_honest_answer_speaks_the_turn_s_language(language) -> None:
    """All locales are equal (AGENTS.md §2) — and none of them says PASS."""
    text = unverified_answer("PASS", missing=("git", "tests"), language=language)
    assert text and not claims_acceptance(text)


def test_an_unknown_language_falls_back_without_crashing() -> None:
    assert unverified_answer("PASS", language="klingon")


# ------------------------------------------------------------- the watch


async def test_only_a_successful_tool_call_becomes_evidence() -> None:
    """A failed git read is not a git read."""
    bus = EventBus()
    trace = uuid4()
    watch = EvidenceWatch(bus, trace)
    watch.start()
    await bus.publish(ActionProposed(trace_id=trace, tool_name="RunCommand", args={"command": "git diff"}))
    await bus.publish(ActionExecuted(trace_id=trace, tool_name="RunCommand", success=False))
    assert not watch.evidence.version_control
    await bus.publish(ActionProposed(trace_id=trace, tool_name="RunCommand", args={"command": "git diff"}))
    await bus.publish(ActionExecuted(trace_id=trace, tool_name="RunCommand", success=True))
    assert watch.evidence.version_control
    assert not watch.evidence.tests
    watch.stop()


async def test_another_turn_s_tool_calls_are_not_this_turn_s_evidence() -> None:
    """A voice turn running alongside must not satisfy a chat turn's gate."""
    bus = EventBus()
    watch = EvidenceWatch(bus, uuid4())
    watch.start()
    other = uuid4()
    await bus.publish(ActionProposed(trace_id=other, tool_name="RunCommand", args={"command": "git diff"}))
    await bus.publish(ActionExecuted(trace_id=other, tool_name="RunCommand", success=True))
    assert not watch.evidence.version_control
    watch.stop()


async def test_a_malformed_event_never_escapes_the_watch() -> None:
    """AP-18: a subscriber exception must not reach the bus's dispatcher."""
    bus = EventBus()
    trace = uuid4()
    watch = EvidenceWatch(bus, trace)
    watch.start()

    class ActionProposed:  # the watch reads the event's type NAME
        tool_name = "RunCommand"

        def __init__(self, trace_id):
            self.trace_id = trace_id

        @property
        def args(self):
            raise RuntimeError("no args for you")

    # The subscriber entry point, as the bus would call it.
    await watch._on_event(ActionProposed(trace))
    assert not watch.evidence.version_control
    watch.stop()
