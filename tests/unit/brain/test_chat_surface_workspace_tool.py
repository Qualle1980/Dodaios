"""The chat surface addresses coding agents through the same service as voice.

Measured 2026-10-29 on the running app: ``GET /api/tools`` (which serves
``BrainManager._tools``, the registry a chat turn offers) listed 139 tools and
**no** ``workspace-orchestrate`` — only the pane-scoped ``agentic-ide-*``
family. The addressed service was substituted into the tool set in exactly one
place, ``BrainSupervisorToolGateway._voice_tools``, which only the Live/voice
catalogue reads.

Everything downstream of the tool already existed and was covered by tests:
project/workspace/agent resolution with a ``request_id``, the cross-project
refusal, ``followthrough`` correlation, ``DelegationResultReady`` carrying
``reply_session_id``, and the server handler that posts a ``coding_result``
notice into the originating ``surface == "jarvis"`` session. The one missing
link was the tool itself on the chat surface, and with it
``current_delegation_origin`` — set in exactly one place
(``jarvis/brain/workspace_tool.py``). Without it ``follow`` is ``None``,
``SessionRegistry.send_prompt`` never arms ``followthrough``, and no result can
reach the conversation that asked for the work.

So these tests pin two properties:

1. Both surfaces derive their coding-agent tools from ONE substitution, so they
   cannot drift apart again.
2. A chat turn offers ``workspace-orchestrate`` and no ambient pane tool. The
   two must not coexist: the pane tools pick a terminal by its place in the
   grid, with no project to address and no result path back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

#: Pane tools that only make sense relative to one open workspace. The two that
#: answer honestly without one (``status``, ``resume``) are deliberately absent.
AMBIENT_PANE_TOOLS = (
    "agentic-ide-prompt",
    "agentic-ide-fanout",
    "agentic-ide-spawn-terminals",
    "agentic-ide-terminal-report",
)


class _FakeWorkspaceGateway:
    """Stands in for the orchestrator; no tool call is made in these tests."""

    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        return {"status": "ok"}


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, args: dict[str, Any], ctx: Any) -> None:  # pragma: no cover
        return None


def _registry() -> dict[str, Any]:
    """A tool registry shaped like the one a chat turn actually offers."""
    names = (*AMBIENT_PANE_TOOLS, "agentic-ide-status", "agentic-ide-resume", "spawn_worker")
    return {name: _FakeTool(name) for name in names}


def _gateway(tools: dict[str, Any]) -> BrainSupervisorToolGateway:
    manager = SimpleNamespace(_tools=tools)
    return BrainSupervisorToolGateway(
        manager, workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspaceGateway())
    )


def test_one_substitution_serves_both_surfaces() -> None:
    """A single helper decides the coding-agent tools, whichever surface asks.

    Before the fix the substitution lived inline in ``_voice_tools`` and the
    chat path had no equivalent, which is exactly how the two surfaces drifted.
    """
    gateway = _gateway(_registry())
    substituted = gateway.with_addressed_workspace_tool(_registry())

    assert WorkspaceOrchestrationTool.name in substituted
    assert not [name for name in substituted if name.startswith("agentic-ide-")]
    # The voice catalogue must be the SAME decision, not a second copy of it.
    voice = {descriptor.name for descriptor in gateway.voice_catalog()}
    assert WorkspaceOrchestrationTool.name in voice
    assert not [name for name in voice if name.startswith("agentic-ide-")]


def test_substitution_keeps_unrelated_tools() -> None:
    """Only the coding-pane family is replaced; nothing else is touched."""
    gateway = _gateway(_registry())
    substituted = gateway.with_addressed_workspace_tool(_registry())
    assert "spawn_worker" in substituted


def test_substitution_without_a_workspace_tool_changes_nothing() -> None:
    """An install whose gateway carries no workspace tool keeps what it had.

    The pane tools are the only coding surface there; removing them would take
    the capability away rather than replace it.
    """
    manager = SimpleNamespace(_tools=_registry())
    bare = BrainSupervisorToolGateway(manager)
    before = _registry()
    assert bare.with_addressed_workspace_tool(before) == before


@pytest.mark.parametrize("pane_tool", AMBIENT_PANE_TOOLS)
def test_a_chat_turn_offers_the_addressed_service_not_an_ambient_pane(
    monkeypatch: pytest.MonkeyPatch, pane_tool: str
) -> None:
    """The regression itself: the chat turn's tool set.

    An ambient pane tool addresses a terminal by its place in the grid. It
    resolves no project, so an order meant for one project can be typed into
    another's agent, and it sets no delegation origin, so the result never
    returns to the conversation that asked.
    """
    from jarvis.brain.manager import BrainManager
    from jarvis.core import runtime_refs

    gateway = _gateway(_registry())
    monkeypatch.setattr(runtime_refs, "get_supervisor_tool_gateway", lambda: gateway)

    offered = BrainManager._offer_addressed_workspace_tool(None, _registry())

    assert WorkspaceOrchestrationTool.name in offered
    assert pane_tool not in offered


def test_a_chat_turn_without_a_gateway_keeps_its_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the runtime registers its gateway a turn must still have tools.

    Registration happens off the boot path, so a turn can arrive first. A gate
    that empties the registry then would blind the brain instead of narrowing
    it — the same defensive stance the neighbouring hide-gates take.
    """
    from jarvis.brain.manager import BrainManager
    from jarvis.core import runtime_refs

    monkeypatch.setattr(runtime_refs, "get_supervisor_tool_gateway", lambda: None)
    before = _registry()
    assert BrainManager._offer_addressed_workspace_tool(None, before) == before


def test_a_failing_gateway_never_blinds_the_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any fault leaves the turn's tools exactly as they were."""
    from jarvis.brain.manager import BrainManager
    from jarvis.core import runtime_refs

    def _boom() -> Any:
        raise RuntimeError("gateway unavailable")

    monkeypatch.setattr(runtime_refs, "get_supervisor_tool_gateway", _boom)
    before = _registry()
    assert BrainManager._offer_addressed_workspace_tool(None, before) == before



def test_addressed_coding_tool_requires_supervisor_first_look() -> None:
    from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

    description = WorkspaceOrchestrationTool.description
    assert "first inspect and read the named project" in description
    assert "complete cheap simple steps directly" in description
    assert "NOT the entire user's original project goal" in description
    assert "independently verify it" in description
    assert "Explicit requests to start a named coding agent" in description
