"""PASS on SHA A is void when the worker later lands SHA B."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeReviewResult, FakeWorkerResult


def _controller(tmp_path: Path) -> tuple[HostController, FakeGitHubProjectStore, FakeHostWorkers]:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    host = HostController(
        store=store,
        workers=workers,
        audit_path=tmp_path / "audit.jsonl",
        max_runs_per_project=20,
    )
    return host, store, workers


def test_pass_on_sha_a_void_when_worker_commits_b(tmp_path: Path) -> None:
    host, store, workers = _controller(tmp_path)
    host.handle(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"tests green"}]}'
    )
    workers.script_assign(
        1,
        FakeWorkerResult(ok=True, commit_sha="aaa111", message="done"),
    )
    host.handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}'
    )
    workers.script_review(
        "aaa111",
        FakeReviewResult(pass_verdict=True, commit_sha="aaa111", findings=[]),
    )
    review = host.handle(
        '{"intent_id":"r1","kind":"REQUEST_REVIEW","project":"demo",'
        '"issue_number":1,"commit_sha":"aaa111"}'
    )
    assert review.ok is True
    assert review.data.get("verdict") == "PASS"

    # Worker pushes a different commit after the PASS.
    store.set_commit_sha("demo", 1, "bbb222")

    accept = host.handle(
        '{"intent_id":"acc1","kind":"ACCEPT_TASK","project":"demo",'
        '"issue_number":1,"commit_sha":"aaa111","tests_green":true}'
    )
    assert accept.ok is False
    assert "sha" in (accept.error or "").lower() or "void" in (accept.error or "").lower()
    issue = store.get_issue("demo", 1)
    assert issue is not None
    assert issue.status != "accepted"
    assert issue.status in {"review", "in-progress", "needs-check"}
