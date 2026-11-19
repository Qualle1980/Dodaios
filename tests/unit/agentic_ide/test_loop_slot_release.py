"""Loop-slot release after create: second start must not stay Queued.

Ultra-review PR #186 task A (2026-11-17): Acquired slots leaked because the
handle was only returned on the result; nobody released it when panes ended
or when launch validation failed after acquire.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from jarvis.agent_chat.loop_scheduler import SCHEDULER, release_terminal_slot
from jarvis.agentic_ide.folders import probe_project
from jarvis.agentic_ide.orchestration import WorkspaceOrchestrator
from jarvis.agentic_ide.session import Session, Terminal
from jarvis.live.state import LiveLedger


class RecordingRegistry:
    def __init__(self, workspace: Session) -> None:
        self._workspace = workspace
        self.workspaces = [workspace]
        self.sessions = [workspace]
        self.active_id = workspace.id
        self.adds: list[dict] = []

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


class QuietSessions:
    def __init__(self, registry: RecordingRegistry) -> None:
        self.registry = registry
        self.sent: list[dict] = []

    async def run(self, args):
        if args.get("action") == "send":
            self.sent.append(dict(args))
            return {"delivery": "accepted", "status": "accepted", "submitted": True}
        return {"delivery": "observed"}


@pytest.fixture
def world(tmp_path):
    from jarvis.agentic_ide import library

    SCHEDULER.reset()
    path = tmp_path / "Personal Jarvis"
    path.mkdir()
    project = library.ensure_project(path)
    workspace = Session(
        id=uuid4().hex,
        folder=str(path),
        name="Personal Jarvis",
        profile=probe_project(path),
        terminals=[],
        created_at=1,
        project_id=project.id,
    )
    registry = RecordingRegistry(workspace)
    sessions = QuietSessions(registry)
    ledger = LiveLedger(tmp_path / "receipts.db")
    orchestrator = WorkspaceOrchestrator(registry, sessions, ledger)
    yield orchestrator, registry, workspace
    ledger.close()
    SCHEDULER.reset()


@pytest.mark.asyncio
async def test_second_create_acquires_after_first_pane_releases_slot(world) -> None:
    orchestrator, registry, workspace = world
    first = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "count": 1,
            "prompt": "Write hello.txt",
        }
    )
    assert first["status"] == "created", first
    handle = first.get("loop_slot_handle") or ""
    assert handle, first
    assert SCHEDULER.status()["slots"]["claude"]["busy"] is True
    owners = [t for t in workspace.terminals if getattr(t, "loop_slot_handle", "")]
    assert owners
    assert owners[0].loop_slot_handle == handle

    # Pane finished (same moment notifications / close would free the slot).
    release_terminal_slot(owners[0])
    assert SCHEDULER.status()["slots"]["claude"]["busy"] is False

    second = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "count": 1,
            "prompt": "Write bye.txt",
        }
    )
    assert second["status"] == "created", second
    assert second.get("status") != "queued"
    assert SCHEDULER.status()["slots"]["claude"]["busy"] is True
    assert second.get("loop_slot_handle")


@pytest.mark.asyncio
async def test_launch_refusal_after_acquire_releases_slot(world) -> None:
    """Acquire then fail between acquire and create — slot must not stay held."""
    orchestrator, _registry, _workspace = world
    refused = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "agents": [{"cli": "not-a-coding-cli-zzzz", "count": 1}],
            "prompt": "Write hello.txt",
        }
    )
    assert refused.get("status") != "created", refused
    assert SCHEDULER.status()["slots"]["claude"]["busy"] is False
    assert SCHEDULER.status()["queue_lengths"]["claude"] == 0
