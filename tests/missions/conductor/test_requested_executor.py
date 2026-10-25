"""A worker the user named is the only executor for that order.

Live 2026-10-25: "Lass Codex nachsehen, wie viele Zeilen ... hat." was
conducted over the whole team, the conductor kept answering from the seat
where it cannot read a file, and the order stalled out without reaching
Codex. These cover the two halves of that: who the user asked for, and the
refusal to call an order done when that worker never ran.
"""
from __future__ import annotations

import pytest

from jarvis.missions.conductor.requested import requested_executor
from jarvis.missions.conductor.shadow import TEAM

LIVE = "Lass Codex nachsehen, wie viele Zeilen jarvis/missions/conductor/loop.py hat."


@pytest.mark.parametrize(("order", "expected"), [
    (LIVE, "codex"),
    ("Lass Codex nachsehen, wie viele Zeilen loop.py hat.", "codex"),  # i18n-allow
    ("Gib das als Auftrag an Codex.", "codex"),  # i18n-allow
    ("Beauftrage Claude Code mit dem Refactoring.", "claude-code"),  # i18n-allow
    ("Codex soll die Datei lesen.", "codex"),  # i18n-allow
    ("Let codex count the lines in loop.py.", "codex"),
    ("Ask Claude Code to review the diff.", "claude-code"),
    ("claude-cli should take this one.", "claude-code"),
])
def test_an_explicit_delegation_names_its_executor(order: str, expected: str) -> None:
    assert requested_executor(order) == expected
    assert expected in TEAM


@pytest.mark.parametrize("order", [
    "Warum funktioniert Codex nicht?",  # i18n-allow: a question about the agent
    "Wie viele Zeilen hat jarvis/missions/conductor/loop.py?",  # i18n-allow
    "Count the lines in loop.py.",
    "Der Codex-Worker ist langsam geworden.",  # i18n-allow
    "",
])
def test_no_explicit_delegation_leaves_routing_alone(order: str) -> None:
    assert requested_executor(order) is None


def test_two_named_agents_are_ambiguous_and_change_nothing() -> None:
    assert requested_executor("Lass Codex und Claude Code nachsehen.") is None  # i18n-allow


def test_the_team_shrinks_to_the_named_worker() -> None:
    """What ``_conduct_mission`` builds: the conductor is no longer an option."""
    requested = requested_executor(LIVE)
    team = {requested: TEAM[requested]} if requested in TEAM else TEAM
    assert list(team) == ["codex"]
    assert "conductor" not in team
