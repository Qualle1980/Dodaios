"""Three failed fix rounds → BLOCKED; no fourth round."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeReviewResult, FakeWorkerResult


def test_three_failed_fix_rounds_block_without_fourth(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    host = HostController(
        store=store,
        workers=workers,
        audit_path=tmp_path / "audit.jsonl",
        max_runs_per_project=20,
    )
    host.handle(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"green"}]}'
    )
    workers.script_assign(1, FakeWorkerResult(ok=True, commit_sha="c1", message="done"))
    host.handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}'
    )

    for round_i in range(1, 4):
        sha = f"c{round_i}"
        store.set_commit_sha("demo", 1, sha)
        workers.script_review(
            sha,
            FakeReviewResult(
                pass_verdict=False,
                commit_sha=sha,
                findings=[f"bug-{round_i}"],
                message="reject",
            ),
        )
        host.handle(
            f'{{"intent_id":"r{round_i}","kind":"REQUEST_REVIEW","project":"demo",'
            f'"issue_number":1,"commit_sha":"{sha}"}}'
        )
        workers.script_fix(
            1,
            FakeWorkerResult(ok=True, commit_sha=f"c{round_i + 1}", message="fixed"),
        )
        result = host.handle(
            f'{{"intent_id":"f{round_i}","kind":"REQUEST_FIX","project":"demo",'
            f'"issue_number":1,"findings":["bug-{round_i}"]}}'
        )
        if round_i < 3:
            assert result.ok is True
            assert store.get_issue("demo", 1).status != "blocked"
        else:
            assert result.ok is False or store.get_issue("demo", 1).status == "blocked"

    issue = store.get_issue("demo", 1)
    assert issue is not None
    assert issue.status == "blocked"
    assert issue.fix_rounds == 3
    fix_calls = [call for call in workers.calls if call.kind == "fix"]
    assert len(fix_calls) == 3

    fourth = host.handle(
        '{"intent_id":"f4","kind":"REQUEST_FIX","project":"demo",'
        '"issue_number":1,"findings":["still broken"]}'
    )
    assert fourth.ok is False
    assert len([call for call in workers.calls if call.kind == "fix"]) == 3
