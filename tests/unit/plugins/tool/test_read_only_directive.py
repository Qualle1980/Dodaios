"""A read order gets the read-only brief, never the production-quality floor.

Live 2026-10-23: "list the TODOs in budget.py, read only" carried "Deliver a
complete, polished, production-quality result" and the worker dumped the file.
"""
from __future__ import annotations

from jarvis.missions.stream_evidence import clean_request_body
from jarvis.plugins.tool.spawn_worker import _build_mission_prompt

_LIVE_ORDER = (
    "Lass Codex die TODO-Kommentare in jarvis/missions/budget.py auflisten. "  # i18n-allow
    "Nur lesen, nichts ändern."  # i18n-allow
)


def test_a_read_order_carries_the_read_only_brief() -> None:
    prompt = _build_mission_prompt(_LIVE_ORDER, action="")

    assert prompt.startswith("READ-ONLY TASK:")
    assert "production-quality" not in prompt
    assert "0 TODOs found" in prompt


def test_a_change_order_keeps_the_quality_floor() -> None:
    prompt = _build_mission_prompt("Erstelle eine Seite wuerfel.html", action="")  # i18n-allow

    assert "production-quality" in prompt
    assert "READ-ONLY TASK" not in prompt


def test_the_read_only_brief_is_stripped_back_off_for_titles() -> None:
    prompt = _build_mission_prompt(_LIVE_ORDER, action="")

    body = clean_request_body(prompt)
    assert "READ-ONLY TASK" not in body
    assert "budget.py" in body


def test_a_read_order_keeps_its_words_and_skips_the_deliverable_push() -> None:
    """The live order lost "Lass Codex die TODO-Kommentare in jarvis/missions"
    because the path segment "missions" read as a routing noun, and it carried
    "use a sensible, complete default deliverable"."""
    prompt = _build_mission_prompt(_LIVE_ORDER, action="")

    assert "TODO-Kommentare in jarvis/missions/budget.py auflisten" in prompt
    assert "complete default deliverable" not in prompt
