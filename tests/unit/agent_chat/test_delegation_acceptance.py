"""What may count as acceptance of delegated coding work, and what may not.

These include the five counter-examples from the independent review of
2026-10-29, every one of which this module used to get wrong: a word inside a
command counted as a test run, ``git status`` counted as reading the change,
and one disclaimer anywhere in the answer licensed a verdict everywhere else.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

import pytest

from jarvis.agent_chat.delegation_acceptance import (
    DIFF_MISSING,
    OBSERVER_ERROR,
    STALE_EVIDENCE,
    TEST_MISSING,
    WRONG_WORKSPACE,
    AcceptanceGate,
    DelegatedJob,
    EvidenceLedger,
    Receipt,
    RecordingTool,
    claims_acceptance,
    gate_for,
    reads_change,
    recording_filter,
    reports_a_pass,
    runs_tests,
    unverified_answer,
)

# Paths are compared, never created: the gate only asks whether one is inside
# the other, which is pure string work on either separator.
ROOT = os.path.join(os.sep + "work", "bubble")
OTHER = os.path.join(os.sep + "work", "other")
HEAD = "abc1234def5678"

COMPLETED_ROW = {
    "task_id": "coding:abc:1:2",
    "status": "completed",
    "report": "Added subtract().",
    "evidence": "Recorded CLI completion and matching report; not independent verification",
    "target": {
        "project_id": "p1",
        "workspace_id": "w1",
        "root": ROOT,
        "head_sha": HEAD,
        "at": 1000.0,
    },
}


def _job(**over) -> DelegatedJob:
    base = {
        "task_id": "coding:abc:1:2",
        "project_id": "p1",
        "workspace_id": "w1",
        "root": ROOT,
        "head_sha": HEAD,
        "at": 1000.0,
    }
    base.update(over)
    return DelegatedJob(**base)


def _ledger(*receipts: Receipt, observer_error: bool = False) -> EvidenceLedger:
    return EvidenceLedger(receipts=list(receipts), observer_error=observer_error)


def _run(command: str, *, ok: bool = True, out: str = "x", cwd: str = ROOT, at: float = 2000.0):
    return Receipt("RunCommand", {"command": command}, ok, out, cwd, at)


def _diff(**over) -> Receipt:
    return _run("git diff", out="diff --git a/calc.py b/calc.py\n+def subtract", **over)


def _tests(**over) -> Receipt:
    return _run("python -m pytest -q", out="2 passed in 0.3s", **over)


# -------------------------------------------------------------- when it arms


def test_an_unverified_completion_arms_a_job_with_its_identity() -> None:
    gate = gate_for([COMPLETED_ROW])
    assert gate is not None
    job = gate.jobs[0]
    assert (job.task_id, job.workspace_id, job.root, job.head_sha) == (
        "coding:abc:1:2",
        "w1",
        ROOT,
        HEAD,
    )


@pytest.mark.parametrize(
    "status", ["needs_input", "superseded", "error", "timed_out", "unverified", "stopped"]
)
def test_a_failed_or_blocked_handback_is_guarded_too(status) -> None:
    """A crashed agent is further from acceptance, not closer to it.

    The first cut armed only on ``completed``, which left the chat free to
    call a blocked job finished.
    """
    gate = gate_for([{**COMPLETED_ROW, "status": status, "evidence": ""}])
    assert gate is not None and gate.jobs[0].status == status


def test_only_an_explicitly_verified_handback_needs_no_gate() -> None:
    """Prose is not a flag.

    Deciding on the ABSENCE of the "not independent verification" marker meant
    a handback that said nothing about itself counted as verified. Nothing
    sets ``verified`` today; it has to be earned where it would be set.
    """
    assert gate_for([{**COMPLETED_ROW, "verified": True}]) is None
    bare = {"task_id": "coding:abc:1:2", "status": "completed", "report": "done"}
    assert gate_for([bare]) is not None
    assert gate_for([{**COMPLETED_ROW, "evidence": "I promise it is fine"}]) is not None


def test_work_that_is_not_a_coding_job_needs_no_gate() -> None:
    assert gate_for([{"task_id": "dispatch:4711", "status": "completed"}]) is None
    assert gate_for([]) is None


# ----------------------------------------------------------- what is a claim


@pytest.mark.parametrize(
    "text",
    [
        "PASS - subtract() is implemented and all tests pass.",
        "All tests pass, we are good.",
        "Verified: the change is in.",
        "Erledigt, alles gr\u00fcn.",  # i18n-allow: German chat output
        "Die Tests laufen durch.",  # i18n-allow: German chat output
        "Gepr\u00fcft und \u00fcbernommen.",  # i18n-allow: German chat output
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
        "T1 is waiting for a permission decision, so nothing is verified.",
        "Der Agent meldet sich fertig, ist aber nicht gepr\u00fcft.",  # i18n-allow: DE output
        "Blocked: the pane needs a decision from you.",
    ],
)
def test_a_report_or_a_disclaimer_is_not_a_claim(text) -> None:
    assert not claims_acceptance(text)


@pytest.mark.parametrize(
    "text",
    [
        "Unverified: not yet verified. PASS, all tests pass.",
        "This is unverified. Anyway: PASS, everything works now.",
        "Die \u00c4nderung ist fertig und fehlerfrei.",  # i18n-allow: German chat output
        "Alles erledigt, l\u00e4uft einwandfrei.",  # i18n-allow: German chat output
        "Done - the implementation is complete and correct.",
    ],
)
def test_a_disclaimer_in_one_clause_does_not_license_a_verdict_in_another(text) -> None:
    """A whole-text disclaimer check let ``Unverified: ... PASS`` straight through."""
    assert claims_acceptance(text)


def test_the_honest_answer_is_never_itself_a_claim() -> None:
    for language in ("en", "de", "es", "pt"):
        text = unverified_answer((DIFF_MISSING, TEST_MISSING), language=language)
        assert text and not claims_acceptance(text), (language, text)


def test_an_unknown_language_falls_back_without_crashing() -> None:
    assert unverified_answer(language="klingon")


def test_the_honest_answer_names_the_actual_gap_and_asks_for_nothing() -> None:
    """It used to say "diff unread" when only the tests were missing, and to
    ask the person to authorize a check they had already asked for."""
    only_tests = unverified_answer((TEST_MISSING,))
    assert "unread" not in only_tests
    assert "test run" in only_tests
    assert "?" not in only_tests


# ------------------------------------------------------- what is a test run


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q",
        "pytest tests/unit",
        "poetry run pytest",
        "uv run pytest -x",
        "npm test",
        "npx vitest run",
        "cargo test",
        "go test ./...",
        "dotnet test",
        r"C:\py\python.exe -m pytest",
    ],
)
def test_a_real_runner_is_a_test_run(command) -> None:
    assert runs_tests("RunCommand", {"command": command})


@pytest.mark.parametrize(
    "command",
    [
        "echo pytest",
        "printf 'pytest passed'",
        'echo "all tests pass"',
        "pytest --collect-only",
        "pytest --help",
        "pytest --version",
        "cat pytest.ini",
        "python -c 'import calc'",
        "ruff check .",
        "git diff",
    ],
)
def test_talking_about_tests_is_not_running_them(command) -> None:
    """A substring match made ``echo pytest`` a test run. Nothing was executed."""
    assert not runs_tests("RunCommand", {"command": command})


# --------------------------------------------------- what is a change read


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git diff --stat",
        "git show HEAD",
        "git -C somewhere diff",
        "git --no-pager diff",
    ],
)
def test_reading_the_diff_is_a_change_read(command) -> None:
    assert reads_change("RunCommand", {"command": command})


@pytest.mark.parametrize(
    "command",
    [
        "git status --porcelain",
        "git status",
        "git rev-parse HEAD",
        "git describe --tags",
        "git log --oneline -1",
        "git commit -m x",
        "git checkout main",
        "echo git diff",
        "ls -la",
    ],
)
def test_asking_where_the_repository_stands_is_not_reading_the_change(command) -> None:
    """``git status`` proves a checkout exists, never what THIS work changed."""
    assert not reads_change("RunCommand", {"command": command})


def test_the_workspace_git_tool_counts_only_for_its_reading_op() -> None:
    assert reads_change("workspace-git", {"op": "state"})
    assert not reads_change("workspace-git", {"op": "commit"})
    assert not reads_change("workspace-git", {})


@pytest.mark.parametrize(
    "output",
    ["2 passed in 0.3s", "ok", "All tests passed", "Test Run Successful", "5 passed, 0 failed"],
)
def test_a_pass_has_to_be_in_the_output(output) -> None:
    assert reports_a_pass(output)


@pytest.mark.parametrize("output", ["", "collected 3 items", "[exit 0]", "E assert False"])
def test_output_without_a_pass_reports_none(output) -> None:
    assert not reports_a_pass(output)


# ---------------------------------------------------- binding it to the job


def test_both_facts_from_the_right_checkout_let_the_answer_stand() -> None:
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("PASS, verified.", _ledger(_diff(), _tests())) is None


@pytest.mark.parametrize("have", ["diff", "tests"])
def test_half_the_evidence_does_not_carry_a_pass(have) -> None:
    """A diff without tests proves something was written, not that it works."""
    ledger = _ledger(_diff() if have == "diff" else _tests())
    gate = AcceptanceGate(jobs=(_job(),))
    verdict = gate.verdict("PASS, all tests pass.", ledger)
    assert verdict is not None and not claims_acceptance(verdict)


def test_a_status_read_plus_an_echo_is_not_acceptance() -> None:
    """The exact pair the review used: both of the old booleans flipped True."""
    ledger = _ledger(_run("git status --porcelain"), _run("echo pytest", out="pytest"))
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("PASS", ledger) is not None
    assert gate.gaps(ledger) == (DIFF_MISSING, TEST_MISSING)


def test_a_check_in_another_workspace_does_not_verify_this_job() -> None:
    ledger = _ledger(_diff(cwd=OTHER), _tests(cwd=OTHER))
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("PASS", ledger) is not None
    assert WRONG_WORKSPACE in gate.gaps(ledger)


def test_the_workspace_git_tool_must_name_this_job_s_workspace() -> None:
    """Omitted means "the visible workspace", which is not provably this one."""
    job = _job()
    elsewhere = Receipt(
        "workspace-git", {"op": "state", "workspace_id": "w2"}, True, "x", "", 2000.0
    )
    unnamed = Receipt("workspace-git", {"op": "state"}, True, "x", "", 2000.0)
    mine = Receipt("workspace-git", {"op": "state", "workspace_id": "w1"}, True, "x", "", 2000.0)
    assert not job.addresses_me(elsewhere)
    assert not job.addresses_me(unnamed)
    assert job.addresses_me(mine)


def test_a_diff_of_the_wrong_commit_does_not_verify() -> None:
    ledger = _ledger(_run("git show 9999999aaaa", out="diff --git a/x"), _tests())
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("PASS", ledger) is not None
    assert WRONG_WORKSPACE in gate.gaps(ledger)


def test_a_diff_of_this_job_s_commit_does_verify() -> None:
    ledger = _ledger(_run(f"git show {HEAD[:7]}", out="diff --git a/calc.py"), _tests())
    assert AcceptanceGate(jobs=(_job(),)).verdict("PASS", ledger) is None


def test_a_test_run_from_before_the_change_does_not_verify_it() -> None:
    ledger = _ledger(_diff(), _tests(at=5.0))
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("PASS", ledger) is not None
    assert STALE_EVIDENCE in gate.gaps(ledger)


def test_a_failed_check_is_not_a_check() -> None:
    ledger = _ledger(_diff(ok=False), _tests(ok=False))
    assert AcceptanceGate(jobs=(_job(),)).verdict("PASS", ledger) is not None


def test_a_diff_command_that_printed_nothing_proves_nothing() -> None:
    """``git diff`` with no output means there was no change to read."""
    ledger = _ledger(_run("git diff", out="[exit 0 - 0.1s]"), _tests())
    assert AcceptanceGate(jobs=(_job(),)).verdict("PASS", ledger) is not None


def test_two_calls_of_the_same_tool_do_not_lend_each_other_their_results() -> None:
    """Receipts carry their own arguments, so there is no pairing to confuse.

    The first cut paired bus events by tool NAME, so a failed ``git diff``
    beside a successful ``echo`` could be credited to either one.
    """
    ledger = _ledger(
        _run("git diff", ok=False, out=""), _run("echo pytest", ok=True, out="pytest")
    )
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.gaps(ledger) == (DIFF_MISSING, TEST_MISSING)


def test_every_waiting_job_needs_its_own_evidence() -> None:
    """Verifying project A does not accept project B's work."""
    a = _job(task_id="coding:a")
    b = _job(task_id="coding:b", workspace_id="w2", root=OTHER)
    gate = AcceptanceGate(jobs=(a, b))
    assert gate.verdict("PASS, both are done.", _ledger(_diff(), _tests())) is not None


# ------------------------------------------------------------ failing closed


def test_a_broken_recorder_withholds_the_claim() -> None:
    """If the gate cannot see what ran, nothing is established."""
    ledger = _ledger(_diff(), _tests(), observer_error=True)
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.gaps(ledger) == (OBSERVER_ERROR,)
    verdict = gate.verdict("PASS", ledger)
    assert verdict is not None and not claims_acceptance(verdict)


def test_an_honest_answer_stands_even_with_a_broken_recorder() -> None:
    """Fail-closed applies to the CLAIM, never to the turn's ability to speak."""
    ledger = _ledger(observer_error=True)
    gate = AcceptanceGate(jobs=(_job(),))
    assert gate.verdict("T1 says it is done; I could not check it.", ledger) is None


# -------------------------------------------------------------- the recorder


class _Tool:
    name = "RunCommand"
    description = "run"
    schema: dict = {"type": "object"}
    risk_tier = "ask"
    read_only = False
    cwd = ROOT

    def __init__(self, *, ok: bool = True, output: str = "2 passed") -> None:
        self._ok = ok
        self._output = output
        self.calls: list[dict] = []

    def read_only_for_args(self, args: dict) -> bool:
        return False

    def describe_args(self, args: dict) -> dict:
        return {"summary": "run"}

    async def execute(self, args, ctx):
        self.calls.append(dict(args))
        from jarvis.core.protocols import ToolResult

        return ToolResult(success=self._ok, output=self._output)


async def test_the_recorder_passes_the_tool_through_untouched() -> None:
    """It may observe. It may not change a tier, a capability or a result."""
    tool = _Tool()
    ledger = EvidenceLedger()
    wrapped = RecordingTool(tool, ledger)
    assert wrapped.name == "RunCommand" and wrapped.risk_tier == "ask"
    assert wrapped.read_only_for_args({}) is False
    assert wrapped.describe_args({}) == {"summary": "run"}

    result = await wrapped.execute({"command": "python -m pytest"}, None)
    assert result.success and result.output == "2 passed"
    assert tool.calls == [{"command": "python -m pytest"}]
    assert ledger.receipts[0].tool == "RunCommand"
    assert ledger.receipts[0].cwd == ROOT
    assert not ledger.observer_error


async def test_a_tool_that_raises_still_raises() -> None:
    class _Boom(_Tool):
        async def execute(self, args, ctx):
            raise RuntimeError("tool failed")

    ledger = EvidenceLedger()
    with pytest.raises(RuntimeError):
        await RecordingTool(_Boom(), ledger).execute({}, None)


async def test_the_filter_wraps_the_whole_surface() -> None:
    ledger = EvidenceLedger()
    tools = {"RunCommand": _Tool(), "Read": _Tool()}
    wrapped = recording_filter(ledger, cwd=ROOT)(tools)
    assert set(wrapped) == {"RunCommand", "Read"}
    await wrapped["RunCommand"].execute({"command": "git diff"}, None)
    assert [r.args["command"] for r in ledger.receipts] == ["git diff"]


def test_an_unrecordable_receipt_fails_closed() -> None:
    ledger = EvidenceLedger()

    class _Hostile(Mapping):
        def keys(self):
            raise RuntimeError("no keys for you")

        def __getitem__(self, key):
            raise RuntimeError("no items either")

        def __iter__(self):
            raise RuntimeError("no iteration either")

        def __len__(self):
            return 1

    ledger.note("RunCommand", _Hostile(), success=True)
    assert ledger.observer_error
    assert AcceptanceGate(jobs=(_job(),)).verdict("PASS", ledger) is not None


def test_a_receipt_is_stamped_when_it_happened() -> None:
    before = time.time()
    ledger = EvidenceLedger()
    ledger.note("RunCommand", {"command": "git diff"}, success=True, output="x", cwd=ROOT)
    assert ledger.receipts[0].at >= before
