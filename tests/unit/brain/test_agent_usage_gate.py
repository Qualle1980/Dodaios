# ruff: noqa: E501 (German user utterances under test are long)
"""Ask-only gate for the agent-usage tool."""

from __future__ import annotations

from jarvis.brain.agent_usage_gate import wants_agent_usage


def test_weekly_limit_question_opens_the_tool() -> None:
    assert wants_agent_usage(
        "Wie viel Prozent meines Wochenlimits in Claude sind verbraucht?"
    )


def test_remaining_quota_question_opens_the_tool() -> None:
    assert wants_agent_usage("How much of my Codex quota is left?")


def test_ordinary_answer_keeps_the_tool_hidden() -> None:
    assert not wants_agent_usage("Was sind die offenen Pull Requests?")  # i18n-allow: German user utterance under test
    assert not wants_agent_usage("Fasse den letzten Commit zusammen.")
    assert not wants_agent_usage("Hallo, wie geht's?")


def test_bare_percent_without_usage_context_stays_closed() -> None:
    assert not wants_agent_usage("Ist 50 Prozent genug Sicherheitspuffer?")
