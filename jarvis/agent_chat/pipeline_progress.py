"""Operator progress lines for Instant's coding pipeline.

Instant reports short status to the operator while a coding task runs:
percentage plus a phase name. Code and constants stay English; the chat line
uses the operator language (German by default for this surface).
"""

from __future__ import annotations

from typing import Final, Literal

Language = Literal["de", "en"]

PHASE_CHECKED_EXISTING: Final[str] = "checked_for_existing_work"
PHASE_BRIEF_WRITTEN: Final[str] = "brief_written"
PHASE_CODING_RUNNING: Final[str] = "coding_agent_running"
PHASE_REVIEW_DONE: Final[str] = "review_done"

#: phase id → (percent, English label, German label)
PIPELINE_PHASES: Final[dict[str, tuple[int, str, str]]] = {
    # i18n-allow: runtime chat output (operator progress)
    PHASE_CHECKED_EXISTING: (
        25,
        "checked for existing work",
        "Bestehende Arbeit geprüft",  # i18n-allow: DE chat output
    ),
    PHASE_BRIEF_WRITTEN: (
        50,
        "brief written",
        "Brief geschrieben",  # i18n-allow: DE chat output
    ),
    PHASE_CODING_RUNNING: (
        75,
        "coding agent running",
        "Coding-Agent läuft",  # i18n-allow: DE chat output
    ),
    PHASE_REVIEW_DONE: (
        100,
        "review done",
        "Review fertig",  # i18n-allow: DE chat output
    ),
}


def phase_percent(phase: str) -> int:
    """Percent for a known phase; raises ``KeyError`` for an unknown id."""
    return PIPELINE_PHASES[phase][0]


def format_operator_progress(phase: str, *, language: Language | str = "de") -> str:
    """One short operator line with percent and phase (German chat by default).

    Unknown phases raise ``KeyError``. Language falls back to German when the
    value is not ``\"en\"``.
    """
    percent, label_en, label_de = PIPELINE_PHASES[phase]
    label = label_en if str(language or "").lower().startswith("en") else label_de
    return f"{percent} % — {label}"


def progress_notice_payload(
    phase: str,
    *,
    turn_id: str,
    language: Language | str = "de",
) -> dict[str, object]:
    """Payload for a ``notice`` event of kind ``pipeline_progress``."""
    percent, label_en, label_de = PIPELINE_PHASES[phase]
    text = format_operator_progress(phase, language=language)
    return {
        "kind": "pipeline_progress",
        "turn_id": turn_id,
        "phase": phase,
        "percent": percent,
        "label_en": label_en,
        "label_de": label_de,
        "text": text,
    }


__all__ = [
    "PHASE_BRIEF_WRITTEN",
    "PHASE_CHECKED_EXISTING",
    "PHASE_CODING_RUNNING",
    "PHASE_REVIEW_DONE",
    "PIPELINE_PHASES",
    "format_operator_progress",
    "phase_percent",
    "progress_notice_payload",
]
