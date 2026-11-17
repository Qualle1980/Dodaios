"""When Instant's correction budget is spent, empty evidence goes to a Codex pane.

Instant keeps the review when its own receipts already carry full evidence.
Otherwise one send goes to a Codex coding pane and the wait loop holds for its
report — the same path delegated coding work already uses. A busy or missing
pane must surface an honest line; nothing is retried in a loop.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agent_chat.delegation_acceptance import Receipt
from jarvis.agent_chat.review_routing import ReviewEvidence
from jarvis.agentic_ide.session import AgentBusyError, SessionError

ROOT = os.path.join(os.sep + "work", "bubble")
ORDER = "Add subtract() to the calculator and run the tests."


def _run(command: str, *, ok: bool = True, out: str = "x", cwd: str = ROOT) -> Receipt:
    return Receipt("RunCommand", {"command": command}, ok, out, cwd)


def _diff(**over: Any) -> Receipt:
    return _run("git diff", out="diff --git a/calc.py b/calc.py\n+def subtract", **over)


def _tests(**over: Any) -> Receipt:
    return _run("python -m pytest -q", out="2 passed in 0.3s", **over)


def _full_evidence() -> ReviewEvidence:
    return ReviewEvidence(
        receipts=(_diff(), _tests()),
        verdict_text="PASS — change read and tests are green.",
    )


@dataclass
class FakeTerminal:
    name: str
    agent: str
    history_id: str = "hist-1"
    submitted: bool = False
    last_submit_at: float | None = None
    status: str = "live"
    submit_generation: int = 1
    process_generation: int = 1
    resume: Any = None
    account: str = ""
    last_prompt: str = ""
    # Non-empty so track_submission's probe returns unverified immediately
    # instead of waiting forever for a local transcript.
    computer_id: str = "fake"


@dataclass
class FakeWorkspace:
    id: str
    project_id: str = "p1"
    folder: str = ROOT
    terminals: list[FakeTerminal] = field(default_factory=list)


@dataclass
class FakeRegistry:
    """Stand-in for the Agentic-IDE registry. Records sends; never opens a PTY."""

    workspaces: list[FakeWorkspace] = field(default_factory=list)
    sends: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    busy: bool = False

    @property
    def sessions(self) -> list[FakeWorkspace]:
        return self.workspaces

    def find_terminal(
        self, wanted: str, workspace_id: str | None = None
    ) -> tuple[FakeWorkspace, FakeTerminal] | None:
        for space in self.workspaces:
            if workspace_id is not None and space.id != workspace_id:
                continue
            for term in space.terminals:
                if (
                    term.name == wanted
                    or term.agent == wanted
                    or wanted == "pane:" + term.history_id
                ):
                    return space, term
        return None

    async def send_prompt(
        self,
        wanted: str,
        text: str,
        *,
        workspace_id: str | None = None,
        when_busy: str = "refuse",
        **kwargs: Any,
    ) -> FakeTerminal:
        if self.busy:
            raise AgentBusyError("The selected coding agent is busy; nothing was sent.")
        found = self.find_terminal(wanted, workspace_id)
        if found is None:
            raise SessionError(f"No terminal called {wanted!r}. Running: none.")
        owner, term = found
        self.sends.append(
            (wanted, text, {"workspace_id": workspace_id or owner.id, "when_busy": when_busy})
        )
        term.submitted = True
        term.last_submit_at = 1.0
        # Mirror Registry.send_prompt: successful delivery is tracked for the wait.
        from jarvis.agentic_ide.delegation_wait import track_submission

        track_submission(self, owner, term)
        return term


def _codex_registry() -> FakeRegistry:
    return FakeRegistry(
        workspaces=[FakeWorkspace("ws-1", terminals=[FakeTerminal("Codex", "codex", "codex-hist")])]
    )


def test_a_full_instant_evidence_skips_codex_when_budget_spent() -> None:
    """(a) Instant already has change+tests+verdict: no Codex send."""
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback

    registry = _codex_registry()

    async def run() -> None:
        result = await attempt_codex_fallback(
            evidence=_full_evidence(),
            prompt="Review the handback.",
            registry=registry,
        )
        assert result.action == "skipped"
        assert registry.sends == []

    asyncio.run(run())


def test_b_empty_evidence_sends_once_to_codex_and_wait_registers() -> None:
    """(b) Empty Instant evidence: one Codex send; wait work is registered."""
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback
    from jarvis.core.delegated_work import DelegatedWork, collect_delegated_work
    from jarvis.core.protocols import ChatTurn, current_chat_turn

    registry = _codex_registry()
    accepted: list[DelegatedWork] = []
    prompt = "Review the coding handback for: " + ORDER
    token = current_chat_turn.set(ChatTurn("s1", "t1", ORDER, True, "trace"))
    try:

        async def run() -> None:
            with collect_delegated_work(accepted.append):
                result = await attempt_codex_fallback(
                    evidence=ReviewEvidence(),
                    prompt=prompt,
                    registry=registry,
                )
            assert result.action == "sent"
            assert len(registry.sends) == 1
            wanted, text, meta = registry.sends[0]
            assert wanted
            assert text == prompt
            assert meta.get("when_busy") == "refuse"
            assert accepted, "Codex submission must be waited for like other delegates"

        asyncio.run(run())
    finally:
        current_chat_turn.reset(token)


def test_c_busy_codex_pane_reports_honestly_without_retry_loop() -> None:
    """(c) AgentBusyError: honest line, one attempt, no silent drop, no loop."""
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback

    registry = _codex_registry()
    registry.busy = True
    calls = {"n": 0}

    async def counting_send(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        raise AgentBusyError("The selected coding agent is busy; nothing was sent.")

    registry.send_prompt = counting_send  # type: ignore[method-assign]

    async def run() -> None:
        result = await attempt_codex_fallback(
            evidence=ReviewEvidence(),
            prompt="Review the handback.",
            registry=registry,
        )
        assert result.action == "busy"
        assert result.message
        assert "busy" in result.message.lower()
        assert calls["n"] == 1
        assert registry.sends == []

    asyncio.run(run())


def test_d_missing_codex_pane_reports_honestly_without_crash() -> None:
    """(d) No Codex pane open: honest line, no exception escapes."""
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback

    registry = FakeRegistry(workspaces=[])

    async def run() -> None:
        result = await attempt_codex_fallback(
            evidence=ReviewEvidence(),
            prompt="Review the handback.",
            registry=registry,
        )
        assert result.action == "missing"
        assert result.message
        assert registry.sends == []

    asyncio.run(run())


def test_budget_spent_wire_in_run_with_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget spent + empty evidence: wait loop sends Codex once and holds for it."""
    from jarvis.agent_chat import codex_fallback, delegation_wait
    from jarvis.agent_chat.events import make_event
    from jarvis.agent_chat.runner_api import TurnHandle
    from jarvis.agent_chat.store import AgentChatStore
    from jarvis.core.delegated_work import DelegatedWork
    from jarvis.core.protocols import ChatTurn, current_chat_turn

    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(delegation_wait, "WAIT_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(delegation_wait, "MAX_CORRECTIONS", 0)

    registry = _codex_registry()
    monkeypatch.setattr(codex_fallback, "default_registry", lambda: registry)

    events: list[dict[str, Any]] = []
    codex_done = asyncio.Event()

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    async def approve(*_a: Any, **_k: Any) -> str:
        return "allow"

    store = AgentChatStore(":memory:")
    session = store.create_session(
        provider="openai", model="", effort="", surface="jarvis", cwd=ROOT
    )
    handle = TurnHandle(
        session=session,
        turn_id="t1",
        emit=emit,
        request_approval=approve,
        cancel=asyncio.Event(),
        output_language="en",
    )
    service = SimpleNamespace(store=store, delegate_wait=None)
    waves = {"n": 0}

    async def probe() -> dict[str, Any] | None:
        if not codex_done.is_set():
            return None
        return {"status": "completed", "report": "Codex verified the diff and tests."}

    async def run_attempt(active: Any, prompt: str) -> None:
        waves["n"] += 1
        if waves["n"] == 1:
            await active.emit(
                make_event(
                    "notice",
                    {
                        "kind": "delegation_correction",
                        "turn_id": active.turn_id,
                        "work": ["coding:1"],
                        "missing": ["no change read", "no test run"],
                        "correcting": ["coding:1"],
                        "left": 0,
                        "receipts": [],
                        "verdict_text": "",
                        "tool_error": False,
                        "codex_fallback": True,
                    },
                )
            )
        await active.emit(
            make_event(
                "turn_finished",
                {"turn_id": active.turn_id, "status": "done", "usage": {}},
            )
        )

    token = current_chat_turn.set(
        ChatTurn(session.session_id, "t1", ORDER, True, "trace")
    )
    try:

        async def run() -> None:
            real = codex_fallback.attempt_codex_fallback

            async def wrapped(**kwargs: Any):
                result = await real(
                    registry=registry,
                    **{k: v for k, v in kwargs.items() if k != "registry"},
                )
                if result.action == "sent":
                    from jarvis.core.delegated_work import register_delegated_work

                    register_delegated_work(
                        DelegatedWork("coding:codex-fallback", "Codex", probe)
                    )
                    codex_done.set()
                return result

            monkeypatch.setattr(codex_fallback, "attempt_codex_fallback", wrapped)
            await delegation_wait.run_with_delegates(service, handle, ORDER, run_attempt)

        asyncio.run(run())
    finally:
        current_chat_turn.reset(token)
        store.close()
    assert registry.sends, "budget spent with empty evidence must send to Codex once"
    assert len(registry.sends) == 1
