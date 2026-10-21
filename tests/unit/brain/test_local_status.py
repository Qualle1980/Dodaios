"""The local status answers: real data in, honest gaps, never an invented fact."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jarvis.brain import local_status
from jarvis.brain.local_status import (
    StatusSources,
    local_status_answer,
    status_kind,
)


def _answer(text: str, lang: str = "de", **kw) -> str | None:
    return asyncio.run(local_status_answer(text, lang, sources=StatusSources(**kw)))


# ---- matching ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Wie ist der Projektstand?", "project_status"),
        ("Jarvis, wie ist der Projektstand?", "project_status"),
        ("What is the project status?", "project_status"),
        ("Läuft Claude noch?", "agents_running"),
        ("Is Codex still running?", "agents_running"),
        ("Gibt es offene Freigaben?", "pending_approvals"),
        ("Any pending approvals?", "pending_approvals"),
        ("Ist alles bereit?", "readiness"),
        ("Ist Jarvis bereit?", "readiness"),
        ("Are you ready?", "readiness"),
    ],
)
def test_status_questions_are_recognised(text: str, kind: str) -> None:
    assert status_kind(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        # A status word inside a real request is NOT a status question: the
        # whole utterance must be one, exactly like the one-shot matcher.
        "Wie ist der Projektstand und erstelle mir dann einen Plan?",
        "Gibt es offene Freigaben, und wenn ja, genehmige sie alle",
        "Erzähl mir etwas Interessantes.",
        "Erkläre mir Python ausführlich.",
        "",
        "   ",
    ],
)
def test_non_status_text_is_not_matched(text: str) -> None:
    assert status_kind(text) is None


def test_a_one_shot_is_never_also_a_status_question() -> None:
    """one_shot_reply owns the greeting/clock set; this module must not steal it."""
    assert status_kind("Danke.") is None
    assert status_kind("Wie spät ist es?") is None


def test_unmatched_text_returns_none_so_the_turn_proceeds() -> None:
    assert _answer("Erzähl mir etwas Interessantes.") is None


def test_unknown_language_falls_through_to_a_model() -> None:
    """A German answer to a Portuguese question is worse than a slow one."""
    assert _answer("Ist alles bereit?", "pt", readiness=_ready({"brain": True})) is None


# ---- honest gaps ------------------------------------------------------------

def _ready(checks: dict[str, bool]):
    async def _call() -> dict[str, bool]:
        return checks
    return _call


def test_an_unwired_source_says_so_instead_of_inventing_a_fact() -> None:
    """The whole point: never answer a status question with a plausible guess."""
    for question in (
        "Läuft Claude noch?",
        "Gibt es offene Freigaben?",
        "Ist alles bereit?",
    ):
        answer = _answer(question)
        assert answer is not None, "a recognised question never reaches a model"
        assert answer == "Das kann ich gerade nicht auslesen."


def test_a_failing_source_reports_unavailable_not_a_default() -> None:
    async def boom() -> list[str]:
        raise RuntimeError("registry is down")

    assert _answer("Läuft Claude noch?", running_agents=boom) == (
        "Das kann ich gerade nicht auslesen."
    )


def test_a_hanging_source_degrades_within_the_deadline(monkeypatch) -> None:
    """A status answer slower than the model turn it replaces is pointless."""
    monkeypatch.setattr(local_status, "SOURCE_TIMEOUT_S", 0.05)

    async def never() -> list[str]:
        await asyncio.sleep(10)
        return []

    assert _answer("Läuft Claude noch?", running_agents=never) == (
        "Das kann ich gerade nicht auslesen."
    )


def test_a_missing_repo_reports_unavailable() -> None:
    assert _answer("Wie ist der Projektstand?") == "Das kann ich gerade nicht auslesen."


def test_a_non_repo_directory_reports_unavailable(tmp_path: Path) -> None:
    """git fails outside a checkout; that is a gap, not "no changes"."""
    assert _answer("Wie ist der Projektstand?", repo=tmp_path) == (
        "Das kann ich gerade nicht auslesen."
    )


# ---- real values ------------------------------------------------------------

def test_running_agents_are_named() -> None:
    async def agents() -> list[str]:
        return ["claude-code", "codex"]

    assert _answer("Läuft Claude noch?", running_agents=agents) == (
        "Aktiv: claude-code, codex."
    )


def test_no_running_agent_is_a_real_answer() -> None:
    async def none() -> list[str]:
        return []

    assert _answer("Läuft Claude noch?", running_agents=none) == "Es läuft kein Agent."


def test_pending_approvals_are_counted() -> None:
    async def two() -> list[object]:
        return [object(), object()]

    assert _answer("Gibt es offene Freigaben?", pending_approvals=two) == (
        "2 offene Freigaben."
    )


def test_no_pending_approvals_is_a_real_answer() -> None:
    async def none() -> list[object]:
        return []

    assert _answer("Gibt es offene Freigaben?", pending_approvals=none) == (
        "Keine offenen Freigaben."
    )


def test_readiness_names_only_what_is_missing() -> None:
    answer = _answer(
        "Ist alles bereit?",
        readiness=_ready({"brain": True, "stt": False, "tts": False}),
    )
    assert answer == "Noch nicht bereit: stt, tts."


def test_readiness_reports_ready_when_everything_passes() -> None:
    assert _answer("Ist alles bereit?", readiness=_ready({"brain": True})) == (
        "Alles bereit."
    )


def test_project_status_reports_the_real_branch_and_dirty_count() -> None:
    """Against this very checkout: the values must come from git, not a guess."""
    answer = _answer("Wie ist der Projektstand?", repo=Path.cwd())
    assert answer is not None
    assert "Das kann ich gerade nicht auslesen." != answer
    assert "Branch " in answer
    # Either phrasing is fine; what matters is that a count was actually read.
    assert "offene Änderungen" in answer or "Keine offenen Änderungen" in answer


def test_english_answers_use_english_lines() -> None:
    async def none() -> list[str]:
        return []

    assert local_status_answer is not None
    answer = asyncio.run(
        local_status_answer(
            "Is Claude still running?", "en", sources=StatusSources(running_agents=none)
        )
    )
    assert answer == "No agent is running."
