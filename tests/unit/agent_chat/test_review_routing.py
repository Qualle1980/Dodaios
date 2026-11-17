"""Who reviews a coding handback: Instant when it has evidence, else Codex.

Slice 1 only: the pure routing decision. The live continuation_prompt wire-up
is a later slice.
"""

from __future__ import annotations

import os

from jarvis.agent_chat.delegation_acceptance import Receipt
from jarvis.agent_chat.review_routing import ReviewEvidence, choose_reviewer

ROOT = os.path.join(os.sep + "work", "bubble")


def _run(command: str, *, ok: bool = True, out: str = "x", cwd: str = ROOT) -> Receipt:
    return Receipt("RunCommand", {"command": command}, ok, out, cwd)


def _diff(**over) -> Receipt:
    return _run("git diff", out="diff --git a/calc.py b/calc.py\n+def subtract", **over)


def _tests(**over) -> Receipt:
    return _run("python -m pytest -q", out="2 passed in 0.3s", **over)


def test_full_evidence_routes_to_instant() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(), _tests()),
        verdict_text="PASS — change read and tests are green.",
    )
    assert choose_reviewer(evidence) == "instant"


def test_missing_test_run_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(),),
        verdict_text="Verified: the diff looks good.",
    )
    assert choose_reviewer(evidence) == "codex"


def test_missing_change_read_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_tests(),),
        verdict_text="All tests pass, we are good.",
    )
    assert choose_reviewer(evidence) == "codex"


def test_instant_tool_error_routes_to_codex() -> None:
    evidence = ReviewEvidence(
        receipts=(_diff(), _tests()),
        verdict_text="PASS — change read and tests are green.",
        tool_error=True,
    )
    assert choose_reviewer(evidence) == "codex"


def test_empty_evidence_routes_to_codex() -> None:
    assert choose_reviewer(None) == "codex"
    assert choose_reviewer(ReviewEvidence()) == "codex"
