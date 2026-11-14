"""Under defer_catalog, high-value tools must be declared directly.

Live sessions open with ``declarations(defer_catalog=True)``
(``jarvis/live/session.py``). Before this fix that path returned only the
meta-surface plus ``computer`` / ``take_appshot``. Memory, wiki, workspace,
search and app-action tools sat in the names-only index and were reachable
solely if the model happened to call ``discover_tools`` — which it often did
not (settled 2026-11-07: GitHub tools → computer instead).

A tool the owner reaches for while working in the app belongs in the direct
set on that surface, in compact form, without blowing the byte budget. What
still does not fit stays discovery-only and is named in the catalog report.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool
from jarvis.core.entry_points import entry_points
from jarvis.live.tools import LiveTools, _PRIORITY_TOOLS

#: Owner-frequent tools that must not depend on the discovery dance.
_MUST_BE_DIRECT = (
    "memory-search",
    "memory-history",
    "memory-topics",
    "search_web",
    "workspace-orchestrate",
    "find-app-action",
    "run-app-action",
    "wiki-recall",
    "show_on_screen",
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


def _bridge() -> LiveTools:
    installed = _registered_tools()
    gateway = BrainSupervisorToolGateway(
        SimpleNamespace(_tools=installed, _config=None),
        workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspace()),
    )
    return LiveTools(
        gateway,
        ledger=None,
        session_id="defer-priority-direct",
        language="en",
        backend_model="gemini-live",
    )


def _declared_canonical(bridge: LiveTools, *, defer: bool) -> set[str]:
    names: set[str] = set()
    for declaration in bridge.declarations(defer_catalog=defer):
        wire = str(declaration.get("name", ""))
        names.add(bridge._names.get(wire, wire))
        names.add(wire)
    return names


@pytest.fixture(scope="module")
def bridge() -> LiveTools:
    return _bridge()


@pytest.mark.parametrize("tool", _MUST_BE_DIRECT)
def test_defer_catalog_declares_owner_frequent_tools_directly(
    tool: str, bridge: LiveTools
) -> None:
    """Red shape: under defer_catalog these were discovery-only (names index only)."""
    voice = {d.name for d in bridge.catalog()}
    if tool not in voice and tool not in _PRIORITY_TOOLS:
        pytest.skip(f"{tool} is not in this environment's voice catalog")
    if tool not in voice:
        pytest.skip(f"{tool} not held in voice catalog for this build")
    declared = _declared_canonical(bridge, defer=True)
    assert tool in declared, (
        f"{tool} is discovery-only under defer_catalog=True: it appears in the "
        "names index but has no direct declaration. A live session opens with "
        "defer_catalog=True, so the model must call discover_tools before it "
        "can use this tool — which is what the owner experiences as unavailable."
    )


def test_defer_catalog_still_keeps_discover_tools(bridge: LiveTools) -> None:
    declared = _declared_canonical(bridge, defer=True)
    assert "discover_tools" in declared
    assert "call_tool" in declared


def test_defer_catalog_reports_what_stays_behind_discovery(
    bridge: LiveTools, caplog: pytest.LogCaptureFixture
) -> None:
    """The #145 loud path must run on the defer surface too."""
    import logging

    with caplog.at_level(logging.INFO, logger="jarvis.live.tools"):
        bridge.declarations(defer_catalog=True)
    joined = "\n".join(r.message for r in caplog.records)
    assert "discover_tools" in joined or "deferred catalog" in joined.lower() or (
        "of" in joined and "tools" in joined
    ), (
        "defer_catalog returned without naming how much of the catalog stays "
        "behind discover_tools (AP-30 / #145 loud path)"
    )
