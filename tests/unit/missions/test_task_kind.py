"""Read, analysis and change orders are told apart from the user's words."""
from __future__ import annotations

import pytest

from jarvis.missions.task_kind import is_read_task, task_kind


@pytest.mark.parametrize(
    ("request_text", "kind"),
    [
        # The live order (2026-10-23).
        ("Lass Codex die TODO-Kommentare in jarvis/missions/budget.py auflisten. "  # i18n-allow
         "Nur lesen, nichts ändern.", "read"),  # i18n-allow
        ("List the TODOs in jarvis/missions/budget.py", "read"),
        ("Find every caller of choose_critic_model", "read"),
        ("Zeig mir, wo der Prüfer das Modell wählt", "read"),  # i18n-allow
        ("Explain how the critic picks its model", "analyze"),
        ("Warum ist der Test in test_routing.py rot?", "analyze"),  # i18n-allow
        ("Review the diff, do not change anything", "analyze"),
        ("Fix the typo in the README", "change"),
        ("Find the bug in budget.py and fix it", "change"),
        ("Erstelle eine HTML-Seite namens wuerfel.html", "change"),  # i18n-allow
        ("Run pytest tests/unit/brain", "change"),
        ("", "change"),
        ("Mach was Schönes", "change"),  # i18n-allow
    ],
)
def test_task_kind(request_text: str, kind: str) -> None:
    assert task_kind(request_text) == kind


def test_research_and_unmarked_analysis_may_still_write_a_report() -> None:
    """Only a read order or an explicitly read-only one forbids files."""
    assert not is_read_task("Explain how the critic picks its model")
    assert not is_read_task("Research the topic drug use in schools")
    assert is_read_task("Explain how the critic picks its model, read only")
    assert is_read_task("List the TODOs in budget.py")


def test_a_read_only_marker_wins_over_a_negated_change_verb() -> None:
    assert is_read_task("Check budget.py without modifying it")
    assert is_read_task("Prüf budget.py, nichts ändern")  # i18n-allow
