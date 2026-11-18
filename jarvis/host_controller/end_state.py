"""Honest end-state wording from host status, never from model prose.

A project is finished only when every planned task is ``accepted``
(:func:`jarvis.host_controller.final_report.all_accepted`). While anything
is still open, user-facing text must say it is still running and give the
host percent — the same discipline as ``action_honesty`` claim families,
but grounded in the project store instead of tool receipts.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from jarvis.host_controller.final_report import all_accepted

log = logging.getLogger(__name__)

# Present-tense / perfect claims that the whole project (or the order) is
# finished. Pattern style mirrors ``jarvis.brain.action_honesty`` claim
# families: normalize, then match; German runtime words stay i18n-allow.
_FINISHED_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:fertig|erledigt|abgeschlossen)\b|"  # i18n-allow: DE matcher
    r"\b(?:alles|auftrag|projekt)\s+(?:ist\s+)?(?:fertig|erledigt|abgeschlossen)\b|"  # i18n-allow
    r"\b(?:done|finished|completed)\b|"
    r"\b(?:project|order|task)\s+(?:is\s+)?(?:done|finished|complete|completed)\b|"
    r"\bproject\s+complete\b|"
    r"\babschlussbericht\b"  # i18n-allow: DE matcher
    r")",
    re.I,
)


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def claims_finished(text: str) -> bool:
    """True when ``text`` claims the order/project is finished."""
    normalized = _normalize(text)
    if not normalized.strip():
        return False
    return _FINISHED_CLAIM_RE.search(normalized) is not None


def percent_int(progress: dict[str, Any] | None) -> int:
    """Whole-number percent from a host progress dict."""
    raw = (progress or {}).get("percent")
    try:
        return int(round(float(raw or 0.0)))
    except (TypeError, ValueError):
        log.warning("host end-state: non-numeric percent %r; using 0", raw)
        return 0


def running_user_phrase(percent: float | int, *, language: str = "de") -> str:
    """User-facing line while the host still has open work."""
    pct = int(round(float(percent or 0)))
    lang = str(language or "de").strip().lower()
    if lang == "de":
        return f"läuft noch, {pct} %"  # i18n-allow: German runtime chat output
    return f"still running, {pct}%"


def finished_user_phrase(*, language: str = "de") -> str:
    """User-facing finished line — only after host acceptance of every task."""
    lang = str(language or "de").strip().lower()
    if lang == "de":
        return "fertig"  # i18n-allow: German runtime chat output
    return "done"


def project_is_finished(progress: dict[str, Any], issues: list[Any]) -> bool:
    """Host truth: every planned task is accepted."""
    return all_accepted(progress, issues)


def user_end_state_phrase(
    progress: dict[str, Any],
    issues: list[Any],
    *,
    language: str = "de",
) -> str:
    """Say done only when the host accepted everything; otherwise an in-progress line."""
    if project_is_finished(progress, issues):
        return finished_user_phrase(language=language)
    return running_user_phrase(percent_int(progress), language=language)


def honest_completion_reply(
    text: str,
    *,
    progress: dict[str, Any],
    issues: list[Any],
    language: str = "de",
) -> str:
    """Rewrite a finished claim when the host still has open tasks.

    When the host has accepted every task, ``text`` is returned unchanged
    (Instant may say done). When the host is not finished and ``text``
    claims completion, the reply becomes the running phrase with percent.
    """
    if project_is_finished(progress, issues):
        return text
    if claims_finished(text):
        phrase = running_user_phrase(percent_int(progress), language=language)
        log.info(
            "host end-state: rewrote finished claim while percent=%s",
            percent_int(progress),
        )
        return phrase
    return text
