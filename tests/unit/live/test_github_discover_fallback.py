"""F-2 — GitHub account questions reach discover/call_tool, not a dead workspace-git end."""

from __future__ import annotations

from jarvis.core.protocols import SupervisorToolDescriptor
from jarvis.live.coordination import WORK_STATUS_RULE
from jarvis.live.discovery import discover


def test_discover_pull_request_returns_github_tools_when_connected() -> None:
    catalog = [
        SupervisorToolDescriptor(
            name="github/list_pull_requests",
            description="List open pull requests in a GitHub repository",
            input_schema={"type": "object"},
            risk_tier="safe",
        ),
        SupervisorToolDescriptor(
            name="github/search_pull_requests",
            description="Search pull requests across repositories",
            input_schema={"type": "object"},
            risk_tier="safe",
        ),
        SupervisorToolDescriptor(
            name="workspace-git",
            description="Git for a coding workspace",
            input_schema={"type": "object"},
            risk_tier="safe",
        ),
        SupervisorToolDescriptor(
            name="gmail/list_messages",
            description="List Gmail messages",
            input_schema={"type": "object"},
            risk_tier="safe",
        ),
    ]
    names = [tool["name"] for tool in discover(catalog, "pull request")["tools"]]
    assert names
    assert any(name.startswith("github/") for name in names)
    assert names[0].startswith("github/")


def test_work_status_rule_routes_account_prs_to_github_plugin() -> None:
    assert "discover_tools" in WORK_STATUS_RULE
    assert "github/*" in WORK_STATUS_RULE or "github/" in WORK_STATUS_RULE
    assert "coding workspace that is open" in WORK_STATUS_RULE
