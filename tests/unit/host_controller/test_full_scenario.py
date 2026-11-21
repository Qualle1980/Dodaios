"""One scenario: PLAN → ASSIGN → REVIEW FAIL → FIX → REVIEW PASS → ACCEPT → 1/3."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.host_controller.instant import handle_instant_text
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeReviewResult, FakeWorkerResult


def test_full_chain_progress_one_of_three(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    run_dir = tmp_path / "run"

    def handle(text: str):
        return handle_instant_text(
            text,
            project="demo",
            store=store,
            workers=workers,
            run_dir=run_dir,
        )

    plan = handle(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"green"},'
        '{"title":"T2","acceptance":"green"},'
        '{"title":"T3","acceptance":"green"}]}'
    )
    assert plan.ok is True
    assert plan.data["issues"] == [1, 2, 3]

    workers.script_assign(
        1,
        FakeWorkerResult(ok=True, commit_sha="shaA", message="done"),
    )
    assign = handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude","allow_subagents":false}'
    )
    assert assign.ok is True
    assert assign.data["commit_sha"] == "shaA"
    assert store.get_issue("demo", 1).commit_sha == "shaA"

    workers.script_review(
        "shaA",
        FakeReviewResult(
            pass_verdict=False,
            commit_sha="shaA",
            findings=["missing test"],
            message="fail",
        ),
    )
    review1 = handle(
        '{"intent_id":"r1","kind":"REQUEST_REVIEW","project":"demo",'
        '"issue_number":1,"commit_sha":"shaA"}'
    )
    assert review1.ok is True
    assert review1.data["verdict"] == "REJECT"
    assert review1.data["findings"] == ["missing test"]

    workers.script_fix(
        1,
        FakeWorkerResult(ok=True, commit_sha="shaB", message="fixed"),
    )
    fix = handle(
        '{"intent_id":"f1","kind":"REQUEST_FIX","project":"demo",'
        '"issue_number":1,"findings":["missing test"]}'
    )
    assert fix.ok is True
    assert fix.data["commit_sha"] == "shaB"
    assert store.get_issue("demo", 1).commit_sha == "shaB"

    workers.script_review(
        "shaB",
        FakeReviewResult(pass_verdict=True, commit_sha="shaB", findings=[], message="pass"),
    )
    review2 = handle(
        '{"intent_id":"r2","kind":"REQUEST_REVIEW","project":"demo",'
        '"issue_number":1,"commit_sha":"shaB"}'
    )
    assert review2.ok is True
    assert review2.data["verdict"] == "PASS"

    accept = handle(
        '{"intent_id":"acc1","kind":"ACCEPT_TASK","project":"demo",'
        '"issue_number":1,"commit_sha":"shaB","tests_green":true}'
    )
    assert accept.ok is True
    assert store.get_issue("demo", 1).status == "accepted"

    progress = handle(
        '{"intent_id":"rp1","kind":"REPORT_PROGRESS","project":"demo"}'
    )
    assert progress.ok is True
    assert progress.data["planned"] == 3
    assert progress.data["accepted"] == 1
    assert progress.data["percent"] == pytest.approx(100.0 / 3.0)

    body = (run_dir / "progress.md").read_text(encoding="utf-8")
    assert "1" in body and "3" in body
    audit_lines = (run_dir / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) >= 6
