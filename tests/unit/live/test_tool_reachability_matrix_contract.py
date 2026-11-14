"""One contract over the four-surface reachability matrix.

If a future change drops an owner-frequent tool from a talk surface, this
file names it. It builds the same offline catalog the matrix script uses
(entry points + voice gateway + shortlist) and asserts invariants — not
fifty per-tool cases.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.tool_shortlist import CORE_PRIORITY, canonical, plan_surface
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool
from jarvis.core.entry_points import entry_points
from jarvis.live.tools import LiveTools, _BUILTIN_TOOLS, _DIRECT_TOOLS, _PRIORITY_TOOLS

#: Tools the owner reaches for while working in the app. Must be direct on
#: live_session (defer_catalog) whenever the voice catalog holds them.
_OWNER_FREQUENT = frozenset(
    {
        "memory-search",
        "memory-history",
        "memory-topics",
        "search_web",
        "wiki-recall",
        "workspace-orchestrate",
        "workspace-git",
        "find-app-action",
        "run-app-action",
        "show_on_screen",
        "remember",
    }
)


class _FakeWorkspace:
    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        return {"status": "ok"}


def _registered_tools() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for entry_point in entry_points(group="jarvis.tool"):
        try:
            tool = entry_point.load()()
        except Exception:  # noqa: BLE001, S112
            continue
        name = str(getattr(tool, "name", "") or "").strip()
        if name:
            tools[name] = tool
    return tools


@pytest.fixture(scope="module")
def bridge() -> LiveTools:
    installed = _registered_tools()
    gateway = BrainSupervisorToolGateway(
        SimpleNamespace(_tools=installed, _config=None),
        workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspace()),
    )
    return LiveTools(
        gateway,
        ledger=None,
        session_id="matrix-contract",
        language="en",
        backend_model="gemini-live",
    )


def _declared(bridge: LiveTools, *, defer: bool) -> set[str]:
    names: set[str] = set()
    for declaration in bridge.declarations(defer_catalog=defer):
        wire = str(declaration.get("name", ""))
        names.add(bridge._names.get(wire, wire))
        names.add(wire)
    return names


def test_matrix_owner_frequent_tools_are_direct_on_live_session(bridge: LiveTools) -> None:
    """live_session opens with defer_catalog=True; these must not be names-only."""
    voice = {d.name for d in bridge.catalog()}
    declared = _declared(bridge, defer=True)
    missing = sorted(
        name for name in _OWNER_FREQUENT if name in voice and name not in declared
    )
    assert not missing, (
        "owner-frequent tools are discovery-only on live_session (defer_catalog): "
        f"{', '.join(missing)}. Declare them directly (compact) or remove them "
        "from _PRIORITY_TOOLS / the voice catalog with an honest reason."
    )


def test_matrix_owner_frequent_tools_stay_in_priority_allow_set() -> None:
    """_PRIORITY_TOOLS is the allow-set for direct-under-defer; keep it explicit."""
    covered = _PRIORITY_TOOLS | _DIRECT_TOOLS | _BUILTIN_TOOLS
    missing = sorted(_OWNER_FREQUENT - covered)
    assert not missing, (
        "an owner-frequent tool is not in _PRIORITY_TOOLS; defer_catalog will "
        f"leave it discovery-only: {', '.join(missing)}"
    )


def test_matrix_workspace_tools_are_core_priority_for_typed_chat() -> None:
    ranked = {canonical(name) for name in CORE_PRIORITY}
    assert canonical("workspace-orchestrate") in ranked
    assert canonical("workspace-git") in ranked


def test_matrix_cli_surface_documents_registry_absence() -> None:
    """Coding CLI seats do not declare BrainManager tools — pin the honest reason."""
    matrix = Path("scripts/tool_reachability_matrix.py").read_text(encoding="utf-8")
    assert "runner_cli" in matrix
    assert "coding CLI seat uses provider tools" in matrix


def test_matrix_chat_shortlist_keeps_workspace_orchestrate_direct(bridge: LiveTools) -> None:
    tools = dict(bridge.gateway._voice_tools())
    if "workspace-orchestrate" not in tools:
        pytest.skip("workspace-orchestrate not in voice tools for this build")
    plan = plan_surface(tools, budget_tokens=6_000, max_direct=0)
    assert "workspace-orchestrate" in plan.direct
