"""Restart mid-task rebuilds from GitHub and does not mark the task done."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeWorkerResult


def test_restart_mid_task_rebuilds_and_needs_check(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    audit = tmp_path / "audit.jsonl"
    host = HostController(
        store=store,
        workers=workers,
        audit_path=audit,
        max_runs_per_project=20,
    )
    host.handle(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"green"}]}'
    )
    workers.script_assign(1, FakeWorkerResult(ok=True, commit_sha=None, message="running"))
    workers.mark_live(1)
    host.handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}'
    )
    issue = store.get_issue("demo", 1)
    assert issue is not None
    assert issue.status == "in-progress"

    # Crash: process gone, new host boots from GitHub snapshot.
    snapshot = store.snapshot()
    workers.clear_live(1)
    store2 = FakeGitHubProjectStore()
    store2.restore(snapshot)
    workers2 = FakeHostWorkers()
    host2 = HostController(
        store=store2,
        workers=workers2,
        audit_path=tmp_path / "audit2.jsonl",
        max_runs_per_project=20,
    )
    recovered = host2.recover()
    issue2 = store2.get_issue("demo", 1)
    assert issue2 is not None
    assert issue2.status == "needs-check"
    assert issue2.status != "accepted"
    assert any(item.get("issue_number") == 1 for item in recovered)
