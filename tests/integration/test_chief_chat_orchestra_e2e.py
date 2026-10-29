"""The typed Jarvis chat is the chief: it looks first, delegates narrowly, verifies.

This is the end-to-end proof for the one claim the policy tests cannot make.
Everything in the chain is the product's own code:

* the real :class:`~jarvis.agent_chat.service.AgentChatService`, its store and
  its event log, on the ``jarvis`` surface;
* the real ``run_with_delegates`` wait loop
  (:mod:`jarvis.agent_chat.delegation_wait`) and the real brain runner
  (:mod:`jarvis.agent_chat.runner_brain`), including the continuation prompt;
* the real surface kit, so the turn's hands are the real folder tools, run
  through a real ``ToolExecutor`` against a real git checkout;
* the real coding handback (:mod:`jarvis.agentic_ide.delegation_wait`), whose
  ``_read`` decides ``completed`` / ``needs_input`` / ``superseded`` from a CLI
  transcript.

Faked, and only here: the chief's judgement (a script —
``tests/fakes/fake_chief_brain.py``), the agent transcript's bytes, and the
pane's send path. Pane *addressing* has its own contract suite
(``tests/contract/test_chat_project_routing.py``); what this file asks is what
happens to the ORIGINAL turn when a coding agent reports back.

No vendor call is made anywhere in this file.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from jarvis.agent_chat import delegation_wait, runner_brain
from jarvis.agent_chat import service as service_mod
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.agentic_ide import agent_transcript
from jarvis.agentic_ide import delegation_wait as ide_wait
from jarvis.core.bus import EventBus
from jarvis.core.config import SafetyBlacklistConfig, SafetyConfig, SafetyWhitelistConfig
from jarvis.core.protocols import ExecutionContext, ToolResult
from jarvis.safety import ApprovalWorkflow, RiskTierEvaluator, ToolExecutor
from tests.fakes.fake_chief_brain import Call, ScriptedChief, Wave

#: This test run's own interpreter — the one that has pytest.
PYTHON = sys.executable.replace("\\", "/")

NO_WORKER_REQUEST = "Zeig den aktuellen Stand von Bubble."  # i18n-allow: the person's words
CONTINUE_REQUEST = "Arbeite an Bubble weiter."  # i18n-allow: the person's words


# ------------------------------------------------------------------ project


def _project(root: Path) -> Path:
    """A real git checkout with one real file, committed."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "README.md").write_text("# Bubble\n\nA tiny bubble.\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    git = ("git", "-C", str(root))
    subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
    subprocess.run([*git, "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "Test"], check=True)
    subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [*git, "commit", "-q", "-m", "init", "--no-gpg-sign"],
        check=True,
        capture_output=True,
        env={"GIT_CONFIG_NOSYSTEM": "1", "HOME": str(root), "PATH": os.environ["PATH"]},
    )
    return root


def _executor(bus: EventBus) -> ToolExecutor:
    safety = SafetyConfig(
        default_tier="safe",
        whitelist=SafetyWhitelistConfig(commands=[]),
        blacklist=SafetyBlacklistConfig(commands=[]),
    )
    return ToolExecutor(bus, RiskTierEvaluator(safety), ApprovalWorkflow(bus))


# ------------------------------------------------------- the coding handback


class FakePane:
    """One Claude pane, as the handback reader sees it."""

    def __init__(self, *, name: str = "T1", prompt: str = "", project: str = "p1") -> None:
        self.history_id = uuid4().hex
        self.name = name
        self.agent = "claude"
        self.account = "default"
        self.computer_id = ""
        self.status = "live"
        self.activity = "running"
        self.submitted = True
        self.last_prompt = prompt
        self.last_submit_at = time.time() - 1
        self.process_generation = 1
        self.resume = type("Resume", (), {"id": "res-" + self.history_id})()
        self.project_id = project


class FakeWorkspace:
    def __init__(self, pane: FakePane, *, project: str = "p1") -> None:
        self.id = uuid4().hex
        self.project_id = project
        self.terminals = [pane]


class FakeRegistry:
    """Just enough registry for the handback's identity checks."""

    def __init__(self, *pairs: tuple[FakeWorkspace, FakePane]) -> None:
        self._pairs = list(pairs)

    def find_terminal(self, terminal_id: str, workspace_id: str) -> tuple[Any, Any] | None:
        for workspace, pane in self._pairs:
            if terminal_id == "pane:" + pane.history_id and workspace_id == workspace.id:
                return (workspace, pane)
        return None


def _transcript(prompt: str, *, reply: str, status: str = "done") -> list[dict[str, Any]]:
    """A CLI transcript that answers exactly this submission."""
    base = int(time.time() * 1000)
    turn = uuid4().hex
    return [
        {"kind": "user_message", "ts_ms": base, "seq": 1, "payload": {"text": prompt}},
        {
            "kind": "assistant_text",
            "ts_ms": base + 10,
            "seq": 2,
            "payload": {"text": reply, "turn_id": turn},
        },
        {
            "kind": "turn_finished",
            "ts_ms": base + 20,
            "seq": 3,
            "payload": {"status": status, "turn_id": turn},
        },
    ]


def _asking(prompt: str) -> list[dict[str, Any]]:
    """A transcript that stopped at a permission question."""
    base = int(time.time() * 1000)
    return [
        {"kind": "user_message", "ts_ms": base, "seq": 1, "payload": {"text": prompt}},
        {
            "kind": "tool_call",
            "ts_ms": base + 5,
            "seq": 2,
            "payload": {
                "call_id": "c1",
                "name": "AskUserQuestion",
                "input": "Do you want to make this edit to calc.py?",
            },
        },
    ]


class DelegateTool:
    """Stands in for the send path of ``workspace-orchestrate``.

    It records the brief the chief composed and arms the product's real
    handback (:func:`jarvis.agentic_ide.delegation_wait.track_submission`) for
    the pane the test wired up. Target *resolution* is not under test here.
    """

    name = "delegate-coding-step"
    description = "Send one bounded coding step to an addressed agent pane."
    risk_tier = "safe"
    read_only = False
    schema = {"type": "object", "properties": {}}

    def __init__(self, registry: FakeRegistry, panes: dict[str, tuple[FakeWorkspace, FakePane]]):
        self._registry = registry
        self._panes = panes
        self.briefs: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        brief = dict(args or {})
        self.briefs.append(brief)
        workspace, pane = self._panes[str(brief.get("terminal_id") or "T1")]
        pane.last_prompt = str(brief.get("prompt") or "")
        pane.last_submit_at = time.time()
        ide_wait.track_submission(self._registry, workspace, pane)
        return ToolResult(success=True, output="accepted")


# ------------------------------------------------------------------ harness


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """A real Jarvis chat whose brain is a scripted chief. Returns a builder."""
    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(service_mod, "_claude_cli_installed", lambda: False)
    monkeypatch.setattr(service_mod, "resolve_runner", lambda *a, **kw: "brain")
    from jarvis.agentic_ide import session as session_mod

    monkeypatch.setattr(session_mod, "account_home", lambda *a, **kw: tmp_path / "home")

    reader: dict[str, Any] = {"fn": lambda *a, **k: []}
    monkeypatch.setattr(
        agent_transcript, "read_events", lambda *a, **k: reader["fn"](*a, **k)
    )

    bus = EventBus()
    project = _project(tmp_path / "bubble")
    store = AgentChatStore(tmp_path / "chat.db")
    svc = AgentChatService(store, assistant_name=lambda: "Jarvis", bus=lambda: bus)

    def build(waves: list[Wave], *, tools: dict[str, Any] | None = None) -> dict[str, Any]:
        chief = ScriptedChief(executor=_executor(bus), waves=waves, tools=tools or {})
        monkeypatch.setattr(runner_brain, "brain_manager", lambda: chief)
        session = svc.create_session(
            provider="openai",
            model="gpt-x",
            effort="",
            cwd=str(project),
            permission_mode="bypass",
            surface="jarvis",
        )
        return {
            "svc": svc,
            "store": store,
            "bus": bus,
            "chief": chief,
            "session": session,
            "project": project,
            "reader": reader,
        }

    return build


async def _run(rig: dict[str, Any], message: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Send one message and wait for the turn to end; return the event log."""
    svc: AgentChatService = rig["svc"]
    sid = rig["session"].session_id
    await svc.send(sid, message)
    run = svc._running.get(sid)
    if run is not None:
        await asyncio.wait_for(asyncio.shield(run.task), timeout)
    return rig["store"].list_events(sid)


def _said(events: list[dict[str, Any]]) -> str:
    """Everything the chat showed the person as an answer, in order."""
    return "\n".join(
        str(e["payload"].get("text") or "") for e in events if e["kind"] == "assistant_text"
    )


def _streamed(events: list[dict[str, Any]]) -> str:
    return "".join(
        str(e["payload"].get("text") or "") for e in events if e["kind"] == "text_delta"
    )


def _finished(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["payload"] for e in events if e["kind"] == "turn_finished"]


# ------------------------------------------------------------------- 1. look


async def test_a_status_question_is_answered_from_the_project_without_any_agent(chat):
    """Case 1: "show me where Bubble stands" reads the project. Nobody is started."""
    delegate = DelegateTool(FakeRegistry(), {})
    rig = chat(
        [
            Wave(
                calls=(
                    Call("Ls", {"path": "."}),
                    Call("Read", {"file_path": "README.md"}),
                ),
                reply="Bubble has calc.py with add(a, b) and a README. Nothing is in flight.",
            )
        ],
        tools={delegate.name: delegate},
    )
    events = await _run(rig, NO_WORKER_REQUEST)

    chief: ScriptedChief = rig["chief"]
    # It really read the real files through the real tools.
    assert [e.tool for e in chief.executed] == ["Ls", "Read"]
    assert all(e.result.success for e in chief.executed), [e.result for e in chief.executed]
    assert "Bubble" in str(chief.executed[1].result.output)
    # One wave, one answer, and no coding agent was addressed at all.
    assert len(chief.calls) == 1
    assert not delegate.briefs
    assert "calc.py" in _said(events)
    assert _finished(events)[-1]["status"] == "done"


# ---------------------------------------------------------------- 2. delegate


async def test_a_hard_step_is_delegated_as_a_bounded_brief_not_the_original_order(chat):
    """Case 2: read first, then send ONE narrow brief — never the person's whole order."""
    pane = FakePane(name="T1")
    workspace = FakeWorkspace(pane)
    registry = FakeRegistry((workspace, pane))
    delegate = DelegateTool(registry, {"T1": (workspace, pane)})
    brief = {
        "project_id": workspace.project_id,
        "workspace_id": workspace.id,
        "terminal_id": "T1",
        "request_id": uuid4().hex,
        "files": ["calc.py"],
        "prompt": "Add subtract(a, b) to calc.py beside add().",
        "done_when": "subtract exists and the module still imports",
    }
    rig = chat(
        [
            Wave(
                calls=(
                    Call("Read", {"file_path": "calc.py"}),
                    Call(delegate.name, brief),
                ),
                reply="I read calc.py and asked T1 for subtract().",
            ),
            Wave(
                calls=(
                    Call("RunCommand", {"command": "git status --porcelain"}),
                    Call("RunCommand", {"command": "git diff"}),
                ),
                reply="Checked: nothing changed on disk, so this is unverified.",
            ),
        ],
        tools={delegate.name: delegate},
    )

    rig["reader"]["fn"] = lambda *a, **k: _transcript(
        pane.last_prompt, reply="Added subtract to calc.py."
    )
    events = await _run(rig, CONTINUE_REQUEST)

    chief: ScriptedChief = rig["chief"]
    assert len(delegate.briefs) == 1, delegate.briefs
    sent = delegate.briefs[0]
    # The child gets identifiers, a file and a finish line — not the chat.
    assert sent["project_id"] and sent["workspace_id"] and sent["terminal_id"]
    assert sent["request_id"] and sent["files"] == ["calc.py"] and sent["done_when"]
    assert CONTINUE_REQUEST not in sent["prompt"]
    # And the original turn came back for a second wave with the report.
    assert len(chief.calls) == 2
    assert "subtract" in chief.executed[-1].system_extra
    assert _finished(events)[-1]["status"] == "done"


# ------------------------------------------------- 3. completion is not PASS


async def test_a_cli_completion_alone_may_not_be_reported_as_a_pass(chat):
    """Case 3 (the gap): the CLI says done, the chief claims PASS, nothing was checked.

    The handback itself is honest — ``_read`` labels it
    ``not independent verification``. The turn must not turn that into a PASS
    for the person, and it must not have STREAMED one either.
    """
    pane = FakePane(name="T1")
    workspace = FakeWorkspace(pane)
    registry = FakeRegistry((workspace, pane))
    delegate = DelegateTool(registry, {"T1": (workspace, pane)})
    rig = chat(
        [
            Wave(
                calls=(Call(delegate.name, {"terminal_id": "T1", "prompt": "Add subtract()."}),),
                reply="Sent to T1.",
            ),
            # No git read. No test run. Just a claim.
            Wave(reply="PASS — subtract() is implemented and all tests pass."),
        ],
        tools={delegate.name: delegate},
    )

    rig["reader"]["fn"] = lambda *a, **k: _transcript(
        pane.last_prompt, reply="Done. PASS, tests pass."
    )
    events = await _run(rig, CONTINUE_REQUEST)

    reports = [
        e["payload"] for e in events if e["kind"] == "notice"
        and e["payload"].get("kind") == "delegation_report"
    ]
    assert reports and reports[0]["status"] == "completed", reports
    answer = _said(events)
    assert "PASS" not in answer.upper(), answer
    assert "PASS" not in _streamed(events).upper(), _streamed(events)


async def test_an_independent_git_and_test_read_is_what_makes_a_pass(chat):
    """Case 4: the chief reads the real diff and runs the real tests — then PASS stands."""
    pane = FakePane(name="T1")
    workspace = FakeWorkspace(pane)
    registry = FakeRegistry((workspace, pane))
    delegate = DelegateTool(registry, {"T1": (workspace, pane)})

    def apply_edit(*_a, **_k):
        # The agent's work really lands on disk before the chief looks.
        path = rig["project"] / "calc.py"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n\ndef subtract(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        tests = rig["project"] / "test_calc.py"
        tests.write_text(
            tests.read_text(encoding="utf-8")
            + "\n\ndef test_subtract():\n    from calc import subtract\n\n"
            "    assert subtract(3, 1) == 2\n",
            encoding="utf-8",
        )
        return _transcript(pane.last_prompt, reply="Added subtract to calc.py.")

    rig = chat(
        [
            Wave(
                calls=(Call(delegate.name, {"terminal_id": "T1", "prompt": "Add subtract()."}),),
                reply="Sent to T1.",
            ),
            Wave(
                calls=(
                    Call("RunCommand", {"command": "git diff"}),
                    # A real suite in the project, on the interpreter that
                    # actually has pytest (PATH python here does not).
                    Call(
                        "RunCommand",
                        # Forward slashes: RunCommand runs through a shell that
                        # would eat the backslashes of a Windows path.
                        {"command": f'"{PYTHON}" -m pytest test_calc.py -q'},
                    ),
                ),
                reply="Verified: the diff adds subtract() and both tests pass. PASS.",
            ),
        ],
        tools={delegate.name: delegate},
    )

    rig["reader"]["fn"] = apply_edit
    events = await _run(rig, CONTINUE_REQUEST)

    chief: ScriptedChief = rig["chief"]
    checks = [e for e in chief.executed if e.tool == "RunCommand"]
    assert len(checks) == 2, checks
    assert all(e.result.success for e in checks), [
        (e.args, e.result.output) for e in checks
    ]
    assert "subtract" in str(checks[0].result.output)
    assert "2 passed" in str(checks[1].result.output), checks[1].result.output
    # Exactly one delegation; the verified answer is allowed to say PASS.
    assert len(delegate.briefs) == 1
    assert len(chief.calls) == 2
    assert "PASS" in _said(events).upper()


# ------------------------------------------------------ 5. refusals and stops


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("asking", "needs_input"),
        ("error", "error"),
    ],
)
async def test_a_blocked_or_failed_agent_is_never_accepted_and_never_retried(
    chat, transcript, expected
):
    """Case 5: a permission question or a crashed turn ends the wait honestly."""
    pane = FakePane(name="T1")
    workspace = FakeWorkspace(pane)
    registry = FakeRegistry((workspace, pane))
    delegate = DelegateTool(registry, {"T1": (workspace, pane)})
    rig = chat(
        [
            Wave(
                calls=(Call(delegate.name, {"terminal_id": "T1", "prompt": "Add subtract()."}),),
                reply="Sent to T1.",
            ),
            Wave(reply="T1 is waiting for a permission decision, so nothing is verified."),
        ],
        tools={delegate.name: delegate},
    )

    def read(*_a, **_k):
        if transcript == "asking":
            return _asking(pane.last_prompt)
        return _transcript(pane.last_prompt, reply="crashed", status="error")

    rig["reader"]["fn"] = read
    events = await _run(rig, CONTINUE_REQUEST)

    reports = [
        e["payload"] for e in events if e["kind"] == "notice"
        and e["payload"].get("kind") == "delegation_report"
    ]
    assert reports and reports[0]["status"] == expected, reports
    # No second brief was sent on suspicion.
    assert len(delegate.briefs) == 1
    assert "PASS" not in _said(events).upper()


# ------------------------------------------------------------- 6. two projects


async def test_two_waiting_projects_keep_their_own_results(chat):
    """Case 6: two panes in two projects report back without crossing over."""
    pane_a = FakePane(name="T1", project="p-a")
    pane_b = FakePane(name="T2", project="p-b")
    ws_a = FakeWorkspace(pane_a, project="p-a")
    ws_b = FakeWorkspace(pane_b, project="p-b")
    registry = FakeRegistry((ws_a, pane_a), (ws_b, pane_b))
    delegate = DelegateTool(registry, {"T1": (ws_a, pane_a), "T2": (ws_b, pane_b)})
    rig = chat(
        [
            Wave(
                calls=(
                    Call(delegate.name, {"terminal_id": "T1", "prompt": "Project A: add sub()."}),
                    Call(delegate.name, {"terminal_id": "T2", "prompt": "Project B: fix typo."}),
                ),
                reply="Both are running.",
            ),
            Wave(reply="A and B reported separately; neither is verified yet."),
        ],
        tools={delegate.name: delegate},
    )

    def read(agent, resume_id, **_k):
        pane = pane_a if resume_id == pane_a.resume.id else pane_b
        return _transcript(pane.last_prompt, reply=f"Finished: {pane.last_prompt}")

    rig["reader"]["fn"] = read
    events = await _run(rig, CONTINUE_REQUEST)

    reports = [
        e["payload"] for e in events if e["kind"] == "notice"
        and e["payload"].get("kind") == "delegation_report"
    ]
    assert len(reports) == 2, reports
    texts = {r["agent_name"]: r["text"] for r in reports}
    assert "Project A" in texts["T1"] and "Project B" not in texts["T1"]
    assert "Project B" in texts["T2"] and "Project A" not in texts["T2"]


# -------------------------------------------------------- 7. the failed probe


async def test_a_pane_whose_model_is_unknown_is_not_a_completed_task(chat):
    """Case 7, offline: the live run's shape — asked for Sonnet, got an edit prompt.

    Nothing was written and the model was never attested. The turn must say
    so, not count the prompt as work.
    """
    pane = FakePane(name="T1")
    workspace = FakeWorkspace(pane)
    registry = FakeRegistry((workspace, pane))
    delegate = DelegateTool(registry, {"T1": (workspace, pane)})
    rig = chat(
        [
            Wave(
                calls=(
                    Call(
                        delegate.name,
                        {"terminal_id": "T1", "prompt": "Add subtract().", "model": "sonnet"},
                    ),
                ),
                reply="Sent to T1 with model=sonnet.",
            ),
            Wave(
                calls=(Call("RunCommand", {"command": "git status --porcelain"}),),
                reply="T1 stopped at a permission prompt and nothing changed. UNVERIFIED.",
            ),
        ],
        tools={delegate.name: delegate},
    )

    rig["reader"]["fn"] = lambda *a, **k: _asking(pane.last_prompt)
    events = await _run(rig, CONTINUE_REQUEST)

    chief: ScriptedChief = rig["chief"]
    # The checkout is genuinely untouched: the status read came back empty.
    status = [e for e in chief.executed if e.tool == "RunCommand"][-1]
    assert "calc.py" not in str(status.result.output), status.result.output
    assert "PASS" not in _said(events).upper()
    assert len(delegate.briefs) == 1
