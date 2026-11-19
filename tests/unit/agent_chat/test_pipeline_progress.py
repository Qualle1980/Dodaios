"""Operator progress lines carry a percentage and a phase name."""

from __future__ import annotations

import pytest

from jarvis.agent_chat.pipeline_progress import (
    PHASE_BRIEF_WRITTEN,
    PHASE_CHECKED_EXISTING,
    PHASE_CODING_RUNNING,
    PHASE_REVIEW_DONE,
    PIPELINE_PHASES,
    format_operator_progress,
    phase_percent,
    progress_notice_payload,
)


def test_phases_are_the_four_pipeline_steps() -> None:
    assert set(PIPELINE_PHASES) == {
        PHASE_CHECKED_EXISTING,
        PHASE_BRIEF_WRITTEN,
        PHASE_CODING_RUNNING,
        PHASE_REVIEW_DONE,
    }
    assert phase_percent(PHASE_CHECKED_EXISTING) == 25
    assert phase_percent(PHASE_BRIEF_WRITTEN) == 50
    assert phase_percent(PHASE_CODING_RUNNING) == 75
    assert phase_percent(PHASE_REVIEW_DONE) == 100


@pytest.mark.parametrize(
    ("phase", "percent", "needle_de", "needle_en"),
    [
        (PHASE_CHECKED_EXISTING, "25 %", "Bestehende Arbeit geprüft", "checked for existing work"),  # i18n-allow
        (PHASE_BRIEF_WRITTEN, "50 %", "Brief geschrieben", "brief written"),
        (PHASE_CODING_RUNNING, "75 %", "Coding-Agent läuft", "coding agent running"),  # i18n-allow
        (PHASE_REVIEW_DONE, "100 %", "Review fertig", "review done"),  # i18n-allow
    ],
)
def test_operator_report_contains_percent_and_phase(
    phase: str, percent: str, needle_de: str, needle_en: str
) -> None:
    de = format_operator_progress(phase, language="de")
    en = format_operator_progress(phase, language="en")
    assert percent in de
    assert needle_de in de
    assert percent in en
    assert needle_en in en


def test_progress_notice_payload_for_chat() -> None:
    payload = progress_notice_payload(
        PHASE_CODING_RUNNING, turn_id="t1", language="de"
    )
    assert payload["kind"] == "pipeline_progress"
    assert payload["turn_id"] == "t1"
    assert payload["phase"] == PHASE_CODING_RUNNING
    assert payload["percent"] == 75
    assert "75 %" in str(payload["text"])
    assert "Coding-Agent läuft" in str(payload["text"])  # i18n-allow


def test_unknown_phase_raises() -> None:
    with pytest.raises(KeyError):
        format_operator_progress("not_a_phase")
