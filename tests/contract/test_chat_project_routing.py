"""Two projects open at once: an order reaches the one it names, or nobody.

The chat surface gained the addressed coding service
(``BrainManager._offer_addressed_workspace_tool``). These are the properties
that make that safe to hand a chat turn, each one asserted against two
projects open simultaneously -- the shape the earlier contract suites did not
exercise together.

The rig from :mod:`tests.contract.test_workspace_orchestration` already opens
"Personal Jarvis" and "Other project", each its own project with its own
workspace and one idle pane, so nothing new is built here.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from tests.contract import test_workspace_orchestration as support

rig = support.rig
runnable = support.runnable


def _two_projects(registry):
    """The two open workspaces, as (A, B) -- distinct projects, distinct panes."""
    first, second = registry.sessions[0], registry.sessions[1]
    assert first.project_id != second.project_id
    assert first.terminals[0].history_id != second.terminals[0].history_id
    return first, second


async def test_an_order_for_project_a_never_reaches_project_b(rig):
    """The whole point: address A explicitly, B must receive nothing."""
    orchestrator, registry, sessions = rig
    a, b = _two_projects(registry)
    # The UI is looking at B, to prove the explicit reference wins.
    registry._active = b.id

    resolved = await support.target(
        rig, project_id=a.project_id, workspace_id=a.id,
        agent="pane:" + a.terminals[0].history_id,
    )
    assert resolved["workspace_id"] == a.id

    receipt = await orchestrator.run({
        "action": "send", **resolved, "request_id": uuid4().hex,
        "prompt": "Rename the greeting helper in this project.",
    })
    assert receipt["status"] == "accepted", receipt

    assert len(sessions.calls) == 1
    assert sessions.calls[0]["workspace_id"] == a.id
    assert sessions.calls[0]["terminal_id"] == "pane:" + a.terminals[0].history_id
    # Nothing was typed into the other project's pane.
    assert b.id not in {call.get("workspace_id") for call in sessions.calls}
    assert "pane:" + b.terminals[0].history_id not in {
        call.get("terminal_id") for call in sessions.calls
    }


async def test_switching_the_visible_project_mid_task_keeps_the_recipient(rig):
    """A resolve is a decision, not a hint about what is on screen.

    The user can open another project while an agent works; the order already
    resolved must still go to the agent it named.
    """
    orchestrator, registry, sessions = rig
    a, b = _two_projects(registry)
    registry._active = a.id

    resolved = await support.target(
        rig, project_id=a.project_id, workspace_id=a.id,
        agent="pane:" + a.terminals[0].history_id,
    )

    # The user switches to the other project before the send lands.
    registry._active = b.id

    receipt = await orchestrator.run({
        "action": "send", **resolved, "request_id": uuid4().hex, "prompt": "Carry on.",
    })
    assert receipt["status"] == "accepted", receipt
    assert sessions.calls[-1]["workspace_id"] == a.id
    assert registry.active_id == b.id  # and the switch itself was not undone


async def test_a_call_sign_resolves_inside_the_named_project(rig):
    """"Tell T2 to ..." keeps working, and stays inside the project named.

    Both projects get a pane called T2, which is the case a call-sign alone
    cannot settle. With the project named, the right T2 must win.
    """
    orchestrator, registry, sessions = rig
    a, b = _two_projects(registry)
    registry._active = b.id

    a_t2 = support.Terminal("t2", "T2", "codex", "Codex", 1, status="live", activity="idle")
    b_t2 = support.Terminal("t2", "T2", "codex", "Codex", 1, status="live", activity="idle")
    a.terminals.append(a_t2)
    b.terminals.append(b_t2)

    resolved = await support.target(
        rig, project_id=a.project_id, workspace_id=a.id, agent="T2",
    )
    assert resolved["workspace_id"] == a.id
    assert resolved["terminal_id"] == "pane:" + a_t2.history_id

    await orchestrator.run({
        "action": "send", **resolved, "request_id": uuid4().hex, "prompt": "Run the tests.",
    })
    assert sessions.calls[-1]["terminal_id"] == "pane:" + a_t2.history_id
    assert sessions.calls[-1]["terminal_id"] != "pane:" + b_t2.history_id


@pytest.mark.parametrize("named", ["zzz-nonexistent-xyz", "Photoshop", "banana"])
async def test_a_project_that_does_not_exist_is_refused_not_guessed(rig, named):
    """An unknown project must never silently become the visible one.

    With two projects open, falling back to whichever is on screen is exactly
    how an order lands in the wrong repository. The names here share no word
    with either open project: the deliberate spoken-name matcher (see
    ``test_spoken_workspace_name_with_filler_words_resolves``) must not be the
    thing under test, only the no-match path.
    """
    orchestrator, registry, sessions = rig
    a, _b = _two_projects(registry)
    registry._active = a.id

    result = await orchestrator.run({"action": "resolve", "project": named})
    assert result["status"] == "needs_clarification", result
    assert not (result.get("target") or {}).get("workspace_id")
    assert not sessions.calls


async def test_an_ambiguous_agent_reference_asks_instead_of_picking(rig):
    """Two panes could match and none is named: ask, never choose for the user."""
    orchestrator, registry, sessions = rig
    a, b = _two_projects(registry)
    a.name = b.name = "Shared name"

    result = await orchestrator.run({"action": "resolve", "workspace": "Shared name"})
    assert result["status"] == "needs_clarification", result
    assert result.get("candidates")
    assert not sessions.calls


async def test_each_project_reports_to_its_own_chat_session(rig, monkeypatch):
    """Two orders from two conversations: each result goes back to its own.

    ``followthrough`` carries the originating ``reply_session_id`` per pane, so
    the two must not cross even when the panes finish together.
    """
    from jarvis.agentic_ide import followthrough

    _orchestrator, registry, _sessions = rig
    a, b = _two_projects(registry)

    monkeypatch.setattr(followthrough, "_snapshot", lambda term: [])

    pending_a = await followthrough.prepare(
        a.terminals[0], "Task A", "Task A",
        {"reply_surface": "chat", "reply_session_id": "chat-a", "lang": "en"},
    )
    pending_b = await followthrough.prepare(
        b.terminals[0], "Task B", "Task B",
        {"reply_surface": "chat", "reply_session_id": "chat-b", "lang": "en"},
    )

    assert pending_a.reply_session_id == "chat-a"
    assert pending_b.reply_session_id == "chat-b"
    assert pending_a.request_id != pending_b.request_id

    followthrough.submitted(a.terminals[0], pending_a)
    followthrough.submitted(b.terminals[0], pending_b)
    assert a.terminals[0].delegation_result.reply_session_id == "chat-a"
    assert b.terminals[0].delegation_result.reply_session_id == "chat-b"


async def test_a_pane_without_a_chat_origin_arms_no_result_path(rig, monkeypatch):
    """An order with no originating conversation must not report into one.

    ``reply_surface`` is supplied by the application, never by a tool argument,
    so an internal or scheduled call cannot promote itself into someone's chat.
    """
    from jarvis.agentic_ide import followthrough
    from jarvis.core.delegation import origin_metadata

    _orchestrator, registry, _sessions = rig
    a, _b = _two_projects(registry)
    monkeypatch.setattr(followthrough, "_snapshot", lambda term: [])

    # No chat turn is in scope: the surface is voice, with no session to answer.
    origin = origin_metadata()
    assert origin["reply_session_id"] == ""

    pending = await followthrough.prepare(a.terminals[0], "Task", "Task", origin)
    assert pending.reply_session_id == ""


def test_the_shared_catalogue_leaves_the_voice_surface_intact() -> None:
    """Voice keeps every tool it had; only the coding family is substituted.

    The chat now calls the same helper, so a regression here would hit both
    surfaces at once -- which is why this asserts the voice catalogue by name.
    """
    from types import SimpleNamespace

    from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
    from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

    class _Gateway:
        async def run(self, args, *, trace_id=""):  # pragma: no cover
            return {"status": "ok"}

    class _Tool:
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.schema = {"type": "object", "properties": {}}

        async def execute(self, args, ctx):  # pragma: no cover
            return None

    registry = {n: _Tool(n) for n in ("agentic-ide-prompt", "spawn_worker", "remember")}
    gateway = BrainSupervisorToolGateway(
        SimpleNamespace(_tools=registry),
        workspace_tool=WorkspaceOrchestrationTool(_Gateway()),
    )
    voice = {descriptor.name for descriptor in gateway.voice_catalog()}

    # The desktop and memory primitives Live relies on are untouched.
    assert {"spawn_worker", "remember"} <= voice
    assert WorkspaceOrchestrationTool.name in voice
    assert not [name for name in voice if name.startswith("agentic-ide-")]
    # And the chat derives the coding tools from the very same decision.
    chat = set(gateway.with_addressed_workspace_tool(registry))
    assert WorkspaceOrchestrationTool.name in chat
    assert not [name for name in chat if name.startswith("agentic-ide-")]
