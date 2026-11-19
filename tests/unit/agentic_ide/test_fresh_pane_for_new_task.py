"""New coding tasks create a fresh pane; busy send returns a create hint.

Slice 1 of the coding loop (2026-11-17). Never send a new task into a busy pane.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from jarvis.agent_chat.loop_scheduler import SCHEDULER
from jarvis.agentic_ide.folders import probe_project
from jarvis.agentic_ide.orchestration import WorkspaceOrchestrator
from jarvis.agentic_ide.session import AgentBusyError, Session, Terminal
from jarvis.live.state import LiveLedger


class RecordingRegistry:
    """Records add_terminal calls; can refuse send when a pane is marked busy."""

    def __init__(self, workspace: Session) -> None:
        self._workspace = workspace
        self.workspaces = [workspace]
        self.sessions = [workspace]
        self.active_id = workspace.id
        self.adds: list[dict] = []
        self.busy_ids: set[str] = set()

    @property
    def session(self):
        return self._workspace

    def get(self, workspace_id: str | None = None):
        if workspace_id in (None, "", self._workspace.id):
            return self._workspace
        return None

    def find_terminal(self, wanted: str, workspace_id: str | None = None):
        for term in self._workspace.terminals:
            pane_id = "pane:" + term.history_id
            if wanted in (term.name, term.agent, pane_id, term.history_id):
                return self._workspace, term
        return None

    async def add_terminal(self, **kwargs):
        self.adds.append(dict(kwargs))
        agent = str(kwargs.get("agent") or "claude")
        name = str(kwargs.get("name") or f"T{len(self._workspace.terminals) + 1}")
        term = Terminal(
            f"t{len(self._workspace.terminals) + 1}",
            name,
            agent,
            agent.title(),
            len(self._workspace.terminals),
            status="live",
            activity="idle",
            activity_at=time.time(),
        )
        self._workspace.terminals.append(term)
        return term

    async def add_terminals(self, count: int, **kwargs):
        created = []
        for _ in range(max(1, int(count))):
            created.append(await self.add_terminal(**kwargs))
        return created, False


class BusyAwareSessions:
    """Mirrors session send: busy pane raises AgentBusyError before any write."""

    def __init__(self, registry: RecordingRegistry) -> None:
        self.registry = registry
        self.sent: list[dict] = []

    async def run(self, args):
        if args.get("action") != "send":
            return {"delivery": "observed"}
        terminal_id = str(args.get("terminal_id") or "")
        found = self.registry.find_terminal(terminal_id, args.get("workspace_id"))
        if found is None:
            return {"delivery": "not_accepted", "status": "stale_target"}
        _owner, term = found
        if term.history_id in self.registry.busy_ids or term.name in self.registry.busy_ids:
            raise AgentBusyError("The selected coding agent is busy; nothing was sent.")
        self.sent.append(dict(args))
        return {"delivery": "accepted", "submitted": True, "completed": False}


@pytest.fixture
def world(tmp_path):
    from jarvis.agentic_ide import library

    SCHEDULER.reset()
    path = tmp_path / "Personal Jarvis"
    path.mkdir()
    project = library.ensure_project(path)
    term = Terminal(
        "t1",
        "Alex",
        "claude",
        "Claude",
        0,
        status="live",
        activity="busy",
        activity_at=time.time(),
    )
    workspace = Session(
        id=uuid4().hex,
        folder=str(path),
        name="Personal Jarvis",
        profile=probe_project(path),
        terminals=[term],
        created_at=1,
        project_id=project.id,
    )
    registry = RecordingRegistry(workspace)
    registry.busy_ids.add(term.history_id)
    sessions = BusyAwareSessions(registry)
    ledger = LiveLedger(tmp_path / "receipts.db")
    orchestrator = WorkspaceOrchestrator(registry, sessions, ledger)
    yield orchestrator, registry, workspace, term, sessions
    ledger.close()
    SCHEDULER.reset()


@pytest.mark.asyncio
async def test_new_coding_task_creates_a_fresh_pane_when_the_existing_one_is_busy(world) -> None:
    orchestrator, registry, workspace, busy_term, sessions = world
    before = len(workspace.terminals)
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "count": 1,
            "prompt": "Add a hello.txt with one line of text.",
        }
    )
    assert result.get("status") in {"created", "queued"} or result.get("success") is not False or (
        result.get("status") == "created"
    )
    # Slot was free (busy pane was not scheduled): create opens a fresh pane.
    assert result["status"] == "created", result
    assert len(workspace.terminals) == before + 1
    assert registry.adds, "new coding task must create a fresh pane"
    new_ids = {t.history_id for t in workspace.terminals} - {busy_term.history_id}
    assert new_ids
    # Nothing was typed into the busy existing pane.
    for send in sessions.sent:
        assert send.get("terminal_id") != "pane:" + busy_term.history_id


@pytest.mark.asyncio
async def test_send_to_busy_pane_returns_a_create_hint(world) -> None:
    orchestrator, registry, workspace, busy_term, sessions = world
    result = await orchestrator.run(
        {
            "action": "send",
            "project_id": workspace.project_id,
            "workspace_id": workspace.id,
            "terminal_id": "pane:" + busy_term.history_id,
            "prompt": "Do a second unrelated task now.",
        }
    )
    assert result.get("busy") is True or result.get("status") == "not_accepted"
    blob = " ".join(
        str(result.get(key) or "") for key in ("next", "reason", "hint", "message")
    ).lower()
    assert "use create after the slot is free" in blob
    assert sessions.sent == []
