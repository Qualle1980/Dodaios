"""Real worker adapters + Codex review SHA binding (scripted drivers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.host_controller.workers import (
    HostWorkerBundle,
    ReviewResult,
    WorkerResult,
)


class ScriptedDriver:
    """Injectable CLI driver. No paid models, no unittest.mock."""

    def __init__(self) -> None:
        self.assign_calls: list[dict] = []
        self.fix_calls: list[dict] = []
        self.review_calls: list[dict] = []
        self.assign_result = WorkerResult(ok=True, commit_sha="aaa111", message="done")
        self.fix_result = WorkerResult(ok=True, commit_sha="bbb222", message="fixed")
        self.review_result = ReviewResult(
            pass_verdict=True,
            commit_sha="aaa111",
            findings=[],
            message="pass",
        )
        self.live: set[int] = set()

    def run_assign(self, **kwargs):  # noqa: ANN003
        self.assign_calls.append(kwargs)
        return self.assign_result

    def run_fix(self, **kwargs):  # noqa: ANN003
        self.fix_calls.append(kwargs)
        return self.fix_result

    def run_review(self, **kwargs):  # noqa: ANN003
        self.review_calls.append(kwargs)
        return self.review_result

    def has_live_process(self, issue_number: int) -> bool:
        return issue_number in self.live


def test_claude_assign_passes_subagent_flags_and_returns_sha(tmp_path: Path) -> None:
    driver = ScriptedDriver()
    driver.assign_result = WorkerResult(
        ok=True,
        commit_sha="cafecafe",
        message="done",
        subagent_kind_used="Explore",
    )
    workers = HostWorkerBundle(
        work_root=tmp_path,
        repo_root=tmp_path,
        drivers={
            "claude": driver.run_assign,
            "grok": driver.run_assign,
            "codex_fix": driver.run_fix,
            "codex_review": driver.run_review,
        },
        live_probe=driver.has_live_process,
    )
    result = workers.assign(
        worker="claude",
        issue_number=3,
        prompt="implement feature",
        allow_subagents=True,
        subagent_kind="Explore",
    )
    assert result.commit_sha == "cafecafe"
    assert result.subagent_kind_used == "Explore"
    assert driver.assign_calls[0]["allow_subagents"] is True
    assert driver.assign_calls[0]["subagent_kind"] == "Explore"
    assert driver.assign_calls[0]["worker"] == "claude"


def test_codex_subagents_stay_off_for_fix(tmp_path: Path) -> None:
    driver = ScriptedDriver()
    workers = HostWorkerBundle(
        work_root=tmp_path,
        repo_root=tmp_path,
        drivers={
            "claude": driver.run_assign,
            "grok": driver.run_assign,
            "codex_fix": driver.run_fix,
            "codex_review": driver.run_review,
        },
        live_probe=driver.has_live_process,
    )
    workers.fix(
        worker="codex",
        issue_number=1,
        prompt="fix it",
        findings=["bug"],
        allow_subagents=True,
        subagent_kind="Explore",
    )
    assert driver.fix_calls[0]["allow_subagents"] is False
    assert driver.fix_calls[0]["subagent_kind"] is None


def test_review_pass_void_when_reviewed_sha_differs_from_recorded(
    tmp_path: Path,
) -> None:
    driver = ScriptedDriver()
    driver.review_result = ReviewResult(
        pass_verdict=True,
        commit_sha="other999",
        findings=[],
        message="pass on wrong sha",
    )
    workers = HostWorkerBundle(
        work_root=tmp_path,
        repo_root=tmp_path,
        drivers={
            "claude": driver.run_assign,
            "grok": driver.run_assign,
            "codex_fix": driver.run_fix,
            "codex_review": driver.run_review,
        },
        live_probe=driver.has_live_process,
        recorded_sha_lookup=lambda _issue: "aaa111",
    )
    review = workers.review(commit_sha="aaa111", issue_number=1)
    assert review.pass_verdict is False
    assert "sha" in " ".join(review.findings).lower() or review.message


def test_review_is_read_only(tmp_path: Path) -> None:
    driver = ScriptedDriver()
    workers = HostWorkerBundle(
        work_root=tmp_path,
        repo_root=tmp_path,
        drivers={
            "claude": driver.run_assign,
            "grok": driver.run_assign,
            "codex_fix": driver.run_fix,
            "codex_review": driver.run_review,
        },
        live_probe=driver.has_live_process,
    )
    workers.review(commit_sha="aaa111", issue_number=2)
    assert driver.review_calls[0].get("read_only") is True
    assert driver.review_calls[0]["commit_sha"] == "aaa111"


@pytest.mark.skipif(
    __import__("os").environ.get("DODAIOS_HOST_LIVE") != "1",
    reason="live workers require DODAIOS_HOST_LIVE=1",
)
def test_live_workers_roundtrip(tmp_path: Path) -> None:
    from jarvis.host_controller.workers import build_live_host_workers

    workers = build_live_host_workers(repo_root=Path("."), work_root=tmp_path)
    # Smoke only: probe construction and live-process map. Paid runs stay off
    # unless the operator explicitly scripts a prompt under the live flag.
    assert workers.has_live_process(999999) is False
