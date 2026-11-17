"""Who reviews a coding handback: Codex after every coding run.

Instant may still record whether its own evidence is complete; that never
skips Codex (Ahmad's rule, 2026-11-17).
"""

from __future__ import annotations

import os

from jarvis.agent_chat.delegation_acceptance import Receipt
from jarvis.agent_chat.review_routing import (
    ReviewEvidence,
    choose_reviewer,
    instant_evidence_complete,
)

ROOT = os.path.join(os.sep + "work", "bubble")


def _run(command: str, *, ok: bool = True, out: str = "x", cwd: str = ROOT) -> Receipt:
    return Receipt("RunCommand", {"command": command}, ok, out, cwd)


def _diff(**over) -> Receipt:
    return _run("git diff", out="diff --git a/calc.py b/calc.py\n+def subtract", **over)


def _tests(**over) -> Receipt:
    return _run("python -m pytest -q", out="2 passed in 0.3s", **over)


def test_full_evidence_still_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(), _tests()),
        verdict_text="PASS — change read and tests are green.",
    )
    assert instant_evidence_complete(evidence) is True
    assert choose_reviewer(evidence) == "codex"


def test_missing_test_run_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(),),
        verdict_text="Verified: the diff looks good.",
    )
    assert instant_evidence_complete(evidence) is False
    assert choose_reviewer(evidence) == "codex"


def test_missing_change_read_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_tests(),),
        verdict_text="All tests pass, we are good.",
    )
    assert instant_evidence_complete(evidence) is False
    assert choose_reviewer(evidence) == "codex"


def test_instant_tool_error_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(), _tests()),
        verdict_text="PASS — change read and tests are green.",
        tool_error=True,
    )
    assert instant_evidence_complete(evidence) is False
    assert choose_reviewer(evidence) == "codex"


def test_empty_evidence_routes_to_codex() -> None:
    assert choose_reviewer(None) == "codex"
    assert choose_reviewer(ReviewEvidence()) == "codex"
    assert instant_evidence_complete(None) is False
    assert instant_evidence_complete(ReviewEvidence()) is False
