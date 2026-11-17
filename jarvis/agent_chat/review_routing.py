"""Decide who reviews a coding handback after Instant tried first.

The General (Instant / ``chatgpt-plan``) always gets the first look and may
still record whether its own receipts are complete. Ahmad's rule (2026-11-17):
Codex reviews after EVERY coding run — Instant evidence completeness never
skips that Codex pass.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from jarvis.agent_chat.delegation_acceptance import (
    Receipt,
    claims_acceptance,
    reads_change,
    reports_a_pass,
    runs_tests,
)

Reviewer = Literal["instant", "codex"]

#: Shell exit marker that RunCommand appends; not real tool output.
_EXIT_MARKER_RE = re.compile(r"\[exit\s+-?\d+\b[^\]]*\]")
_GIT_TOOL = "workspace-git"


@dataclass(frozen=True)
class ReviewEvidence:
    """What Instant produced when asked to review a coding handback.

    ``tool_error`` covers an Instant failure or a broken observer: nothing
    Instant reported can be trusted, so Codex must review.
    """

    receipts: tuple[Receipt, ...] = ()
    verdict_text: str = ""
    tool_error: bool = False


def _said_something(output: str) -> bool:
    """Output that is more than the shell's own exit marker.

    Same rule as :mod:`jarvis.agent_chat.delegation_acceptance` uses for a
    change read; kept local because that helper is private there.
    """
    return bool(_EXIT_MARKER_RE.sub("", str(output or "")).strip())


def _has_change_read(receipts: Iterable[Receipt]) -> bool:
    for receipt in receipts:
        if not receipt.success:
            continue
        if not reads_change(receipt.tool, receipt.args):
            continue
        if receipt.tool == _GIT_TOOL or _said_something(receipt.output):
            return True
    return False


def _has_test_run(receipts: Iterable[Receipt]) -> bool:
    for receipt in receipts:
        if not receipt.success:
            continue
        if not runs_tests(receipt.tool, receipt.args):
            continue
        if reports_a_pass(receipt.output):
            return True
    return False


def instant_evidence_complete(evidence: ReviewEvidence | None) -> bool:
    """True when Instant's own receipts already carry change + tests + verdict.

    Recorded for honesty and status; it does not skip Codex review.
    """
    if evidence is None or evidence.tool_error:
        return False
    if not _has_change_read(evidence.receipts):
        return False
    if not _has_test_run(evidence.receipts):
        return False
    if not claims_acceptance(evidence.verdict_text):
        return False
    return True


def choose_reviewer(_evidence: ReviewEvidence | None) -> Reviewer:
    """Always Codex after a coding run.

    ``instant_evidence_complete`` still reports whether Instant already had
    full evidence; that fact is never a skip.
    """
    return "codex"


__all__ = [
    "ReviewEvidence",
    "Reviewer",
    "choose_reviewer",
    "instant_evidence_complete",
]
