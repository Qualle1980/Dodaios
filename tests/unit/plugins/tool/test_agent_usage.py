"""Tests for the agent-usage router tool (fakes only, no unittest.mock)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from jarvis.agent_usage import AccountUsage, UsageWindow
from jarvis.core.protocols import ExecutionContext
from jarvis.plugins.tool.agent_usage import AgentUsageTool, format_usage_report


@dataclass(frozen=True)
class _FakeAccount:
    id: str
    platform: str
    label: str = "default"
    config_dir: Path = Path(".")
    builtin: bool = True


def _ctx() -> ExecutionContext:
    return ExecutionContext(
        trace_id=uuid4(), user_utterance="", config={}, memory_read=None
    )


@pytest.mark.asyncio
async def test_tool_returns_percents_from_collect() -> None:
    account = _FakeAccount(id="claude:default", platform="claude")
    reading = AccountUsage(
        account_id=account.id,
        platform="claude",
        status="ok",
        source="live",
        windows=(
            UsageWindow(kind="session", percent=40.0, severity="normal", window_minutes=300),
            UsageWindow(kind="weekly", percent=72.0, severity="normal", window_minutes=10080),
        ),
    )

    def list_accounts() -> list[_FakeAccount]:
        return [account]

    def collect(accounts, *, refresh: bool = False):  # noqa: ARG001
        assert list(accounts) == [account]
        return {account.id: reading}

    tool = AgentUsageTool(list_accounts_fn=list_accounts, collect_fn=collect)
    result = await tool.execute({"platform": "claude"}, _ctx())
    assert result.success is True
    text = str(result.output)
    assert "claude:default" in text
    assert "source=live" in text
    assert "40% used" in text
    assert "60% remaining" in text
    assert "72% used" in text
    assert "28% remaining" in text


@pytest.mark.asyncio
async def test_cached_source_is_labelled() -> None:
    account = _FakeAccount(id="codex:default", platform="codex")
    reading = AccountUsage(
        account_id=account.id,
        platform="codex",
        status="ok",
        source="cached",
        windows=(UsageWindow(kind="weekly", percent=10.0, severity="normal"),),
    )
    tool = AgentUsageTool(
        list_accounts_fn=lambda: [account],
        collect_fn=lambda accounts, *, refresh=False: {account.id: reading},
    )
    result = await tool.execute({}, _ctx())
    assert result.success is True
    assert "source=cached" in str(result.output)


def test_format_report_marks_unavailable_without_inventing_percents() -> None:
    reading = AccountUsage(
        account_id="grok-build:default",
        platform="grok-build",
        status="unavailable",
        message="Usage could not be read for this seat.",
    )
    text = format_usage_report({"grok-build:default": reading})
    assert "unavailable" in text
    assert "%" not in text


def test_tool_is_in_router_tools() -> None:
    from jarvis.brain.factory import ROUTER_TOOLS

    assert "agent-usage" in ROUTER_TOOLS


def test_factory_registers_agent_usage_without_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worktree DevApp: importlib.metadata may lack the new entry point."""
    from jarvis.brain import factory as factory_mod
    from jarvis.core import entry_points as ep_mod

    monkeypatch.setattr(ep_mod, "entry_points", lambda *, group: ())
    tools = factory_mod._load_tools_for_tier(
        "router",
        bus=None,
        executor=None,
        harness_manager=None,
        user_profile=None,
        people=None,
        config=type("Cfg", (), {"harness": type("H", (), {"max_output_chars": 1000})()})(),
    )
    assert "agent-usage" in tools
    assert getattr(tools["agent-usage"], "name", "") == "agent-usage"
