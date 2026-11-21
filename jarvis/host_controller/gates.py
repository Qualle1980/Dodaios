"""Host-side gates. Never trust a worker or Instant claim alone.

Reuses the AcceptanceGate idea from ``jarvis.agent_chat.delegation_acceptance``:
a CLI "done" is delivery, not a PASS. Here the host also binds PASS to an
exact commit SHA and caps fix rounds at ``MAX_FIX_ROUNDS`` (same constant
discipline as ``MAX_CRITIC_LOOPS`` in ADR-0009).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.host_controller.types import MAX_FIX_ROUNDS


@dataclass
class ReviewRecord:
    issue_number: int
    commit_sha: str
    pass_verdict: bool
    findings: list[str] = field(default_factory=list)


@dataclass
class GateBook:
    """In-process evidence the host itself recorded (rebuilt empty on restart;
    GitHub status remains authoritative for task state)."""

    reviews: dict[tuple[str, int, str], ReviewRecord] = field(default_factory=dict)

    def record_review(
        self,
        project: str,
        issue_number: int,
        *,
        commit_sha: str,
        pass_verdict: bool,
        findings: list[str] | None = None,
    ) -> ReviewRecord:
        record = ReviewRecord(
            issue_number=issue_number,
            commit_sha=commit_sha,
            pass_verdict=pass_verdict,
            findings=list(findings or []),
        )
        self.reviews[(project, issue_number, commit_sha)] = record
        return record

    def pass_for_sha(self, project: str, issue_number: int, commit_sha: str) -> bool:
        record = self.reviews.get((project, issue_number, commit_sha))
        return bool(record and record.pass_verdict)


def require_commit_for_review(commit_sha: str | None) -> str:
    sha = (commit_sha or "").strip()
    if not sha:
        raise ValueError("worker done without a commit is not a review request")
    return sha


def sha_still_matches(recorded_sha: str | None, claimed_sha: str) -> bool:
    """A PASS is void when the task's current commit is no longer that SHA."""
    current = (recorded_sha or "").strip()
    claimed = (claimed_sha or "").strip()
    return bool(current) and current == claimed


def can_accept(
    *,
    gate_pass: bool,
    current_sha: str | None,
    claimed_sha: str,
    tests_green: bool,
    merge_allowed: bool = True,
) -> tuple[bool, str]:
    if not tests_green:
        return False, "required tests are not green; failing check is never bypassed"
    if not sha_still_matches(current_sha, claimed_sha):
        return False, "PASS is void: commit SHA no longer matches; new review required"
    if not gate_pass:
        return False, "host has no PASS on this exact commit SHA"
    if not merge_allowed:
        return False, "merge policy does not allow acceptance"
    return True, "ok"


def next_fix_round_or_block(current_rounds: int) -> tuple[bool, int, str]:
    """Return (allowed, new_round_count, reason). Cap is MAX_FIX_ROUNDS."""
    if current_rounds >= MAX_FIX_ROUNDS:
        return False, current_rounds, f"fix round cap {MAX_FIX_ROUNDS} reached; BLOCKED"
    return True, current_rounds + 1, "ok"
