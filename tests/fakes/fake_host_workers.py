"""Fake Claude/Grok workers and Codex reviewer for host-controller tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class QuotaExhausted(Exception):
    """Provider reported its subscription/window is spent."""


@dataclass
class FakeDispatchRecord:
    worker: str
    issue_number: int
    prompt: str
    allow_subagents: bool
    subagent_kind: str | None
    kind: str  # assign | fix | review


@dataclass
class FakeWorkerResult:
    ok: bool
    commit_sha: str | None = None
    message: str = ""
    quota_exhausted: bool = False
    subagent_kind_used: str | None = None


@dataclass
class FakeReviewResult:
    pass_verdict: bool
    commit_sha: str
    findings: list[str] = field(default_factory=list)
    message: str = ""


class FakeHostWorkers:
    """Records dispatch and returns scripted worker/reviewer outcomes."""

    def __init__(self) -> None:
        self.calls: list[FakeDispatchRecord] = []
        self.assign_results: dict[int, FakeWorkerResult] = {}
        self.fix_results: dict[int, list[FakeWorkerResult]] = {}
        self.review_results: dict[str, FakeReviewResult] = {}
        self.paid_fallback_attempted = False
        self.live_processes: set[int] = set()

    def script_assign(self, issue_number: int, result: FakeWorkerResult) -> None:
        self.assign_results[issue_number] = result

    def script_fix(self, issue_number: int, result: FakeWorkerResult) -> None:
        self.fix_results.setdefault(issue_number, []).append(result)

    def script_review(self, commit_sha: str, result: FakeReviewResult) -> None:
        self.review_results[commit_sha] = result

    def mark_live(self, issue_number: int) -> None:
        self.live_processes.add(issue_number)

    def clear_live(self, issue_number: int) -> None:
        self.live_processes.discard(issue_number)

    def has_live_process(self, issue_number: int) -> bool:
        return issue_number in self.live_processes

    def assign(
        self,
        *,
        worker: str,
        issue_number: int,
        prompt: str,
        allow_subagents: bool = False,
        subagent_kind: str | None = None,
    ) -> FakeWorkerResult:
        self.calls.append(
            FakeDispatchRecord(
                worker=worker,
                issue_number=issue_number,
                prompt=prompt,
                allow_subagents=allow_subagents,
                subagent_kind=subagent_kind,
                kind="assign",
            )
        )
        result = self.assign_results.get(
            issue_number,
            FakeWorkerResult(ok=True, commit_sha="deadbeef", message="done"),
        )
        if result.quota_exhausted:
            raise QuotaExhausted(result.message or "quota exhausted")
        return result

    def fix(
        self,
        *,
        worker: str,
        issue_number: int,
        prompt: str,
        findings: list[str],
        allow_subagents: bool = False,
        subagent_kind: str | None = None,
    ) -> FakeWorkerResult:
        self.calls.append(
            FakeDispatchRecord(
                worker=worker,
                issue_number=issue_number,
                prompt=f"{prompt}\nFINDINGS:{findings}",
                allow_subagents=allow_subagents,
                subagent_kind=subagent_kind,
                kind="fix",
            )
        )
        queue = self.fix_results.get(issue_number) or []
        if queue:
            result = queue.pop(0)
        else:
            result = FakeWorkerResult(ok=False, commit_sha=None, message="fix failed")
        if result.quota_exhausted:
            raise QuotaExhausted(result.message or "quota exhausted")
        return result

    def review(
        self, *, commit_sha: str, issue_number: int, acceptance: str = ""
    ) -> FakeReviewResult:
        self.calls.append(
            FakeDispatchRecord(
                worker="codex",
                issue_number=issue_number,
                prompt=f"review {commit_sha}",
                allow_subagents=False,
                subagent_kind=None,
                kind="review",
            )
        )
        result = self.review_results.get(commit_sha)
        if result is None:
            return FakeReviewResult(
                pass_verdict=False,
                commit_sha=commit_sha,
                findings=["no scripted review"],
                message="missing review",
            )
        return result

    def attempt_paid_fallback(self, *_args: Any, **_kwargs: Any) -> None:
        """Must never be called by the host on quota exhaustion."""
        self.paid_fallback_attempted = True
        raise AssertionError("paid fallback must not run on quota exhaustion")
