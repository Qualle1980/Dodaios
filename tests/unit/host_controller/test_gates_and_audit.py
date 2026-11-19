"""Host gates and append-only audit log."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeWorkerResult


def test_worker_done_without_commit_is_not_a_review_request(tmp_path: Path) -> None:
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
    workers.script_assign(
        1,
        FakeWorkerResult(ok=True, commit_sha=None, message="done"),
    )
    host.handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}'
    )
    review = host.handle(
        '{"intent_id":"r1","kind":"REQUEST_REVIEW","project":"demo",'
        '"issue_number":1,"commit_sha":""}'
    )
    assert review.ok is False
    assert "commit" in (review.error or "").lower()
    review_calls = [call for call in workers.calls if call.kind == "review"]
    assert review_calls == []


def test_failing_check_never_bypassed_on_accept(tmp_path: Path) -> None:
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
    store.set_status("demo", 1, "review")
    store.set_commit_sha("demo", 1, "abc123")
    # Host has no PASS recorded for this SHA.
    accept = host.handle(
        '{"intent_id":"acc1","kind":"ACCEPT_TASK","project":"demo",'
        '"issue_number":1,"commit_sha":"abc123","tests_green":false}'
    )
    assert accept.ok is False
    assert store.get_issue("demo", 1).status != "accepted"


def test_every_intent_is_audited(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    audit_path = tmp_path / "audit.jsonl"
    host = HostController(
        store=store,
        workers=workers,
        audit_path=audit_path,
        max_runs_per_project=20,
    )
    host.handle(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"green"}]}'
    )
    host.handle('{"intent_id":"rp1","kind":"REPORT_PROGRESS","project":"demo"}')
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    rows = [json.loads(line) for line in lines]
    assert any(row.get("intent_id") == "p1" for row in rows)
    assert all("ts" in row for row in rows)
    assert all("action" in row for row in rows)
