"""Coding workspace tools must be direct on the typed-chat shortlist.

``workspace-orchestrate`` and ``workspace-git`` are how the owner hands work
to coding agents from the app. Being callable behind ``find_tools`` is not
the same as being offered: a model that does not discover them first looks
like the capability is missing. They belong in ``CORE_PRIORITY`` so
``plan_surface`` keeps them in the direct set under the normal token budget.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.tool_shortlist import CORE_PRIORITY, canonical, plan_surface
from jarvis.brain.workspace_git_tool import WorkspaceGitTool
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool


class _FakeWorkspace:
    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        return {"status": "ok"}


class _FakeTool:
    def __init__(self, name: str, *, heavy: bool = False) -> None:
        self.name = name
        self.description = "x" * (800 if heavy else 40)
        props = {
            f"arg{i}": {"type": "string", "description": "y" * 20}
            for i in range(30 if heavy else 1)
        }
        self.schema = {"type": "object", "properties": props}
        self.is_mcp_tool = False

    async def execute(self, args: dict[str, Any], ctx: Any) -> None:  # pragma: no cover
        return None


def test_workspace_tools_are_named_in_core_priority() -> None:
    ranked = {canonical(name) for name in CORE_PRIORITY}
    assert canonical("workspace-orchestrate") in ranked, (
        "workspace-orchestrate missing from CORE_PRIORITY — chat shortlist "
        "will defer it behind find_tools"
    )
    assert canonical("workspace-git") in ranked, (
        "workspace-git missing from CORE_PRIORITY — chat shortlist will "
        "defer it behind find_tools"
    )


def test_workspace_tools_stay_direct_under_the_chat_budget() -> None:
    """Crowd the registry the way a connected install does; workspace stays direct."""
    gateway = BrainSupervisorToolGateway(
        SimpleNamespace(_tools={}, _config=None),
        workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspace()),
    )
    tools: dict[str, Any] = {
        f"bulk-{i}": _FakeTool(f"bulk-{i}", heavy=True) for i in range(40)
    }
    tools["search_web"] = _FakeTool("search_web")
    tools = gateway.with_addressed_workspace_tool(tools)
    # Mirror BrainManager's default tool-surface budget.
    plan = plan_surface(tools, budget_tokens=6_000, max_direct=0)
    assert WorkspaceOrchestrationTool.name in plan.direct, (
        f"workspace-orchestrate was deferred behind find_tools; "
        f"direct={sorted(plan.direct)[:12]} deferred={sorted(plan.deferred)[:12]}"
    )
    assert WorkspaceGitTool.name in plan.direct, (
        f"workspace-git was deferred behind find_tools; "
        f"direct={sorted(plan.direct)[:12]} deferred={sorted(plan.deferred)[:12]}"
    )
