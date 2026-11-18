"""Structural ask-only hide for agent-usage on ordinary turns."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis.brain.manager import BrainManager


def test_ordinary_turn_hides_agent_usage_tool() -> None:
    tools = {
        "search-web": SimpleNamespace(name="search-web"),
        "agent-usage": SimpleNamespace(name="agent-usage"),
    }
    out = BrainManager._hide_agent_usage_tool_without_request(
        SimpleNamespace(), tools, "Fasse den letzten Commit zusammen."
    )
    assert "agent-usage" not in out
    assert "search-web" in out


def test_usage_question_keeps_agent_usage_tool() -> None:
    tools = {
        "search-web": SimpleNamespace(name="search-web"),
        "agent-usage": SimpleNamespace(name="agent-usage"),
    }
    out = BrainManager._hide_agent_usage_tool_without_request(
        SimpleNamespace(),
        tools,
        "Wie viel Prozent meines Wochenlimits in Claude sind verbraucht?",
    )
    assert "agent-usage" in out
