"""Quota exhausted → waiting-quota; never a silent paid-key fallback."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeWorkerResult


def test_quota_exhausted_waits_without_paid_fallback(tmp_path: Path) -> None:
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
        FakeWorkerResult(ok=False, quota_exhausted=True, message="quota exhausted"),
    )
    result = host.handle(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}'
    )
    assert result.ok is False or store.get_issue("demo", 1).status == "waiting-quota"
    issue = store.get_issue("demo", 1)
    assert issue is not None
    assert issue.status == "waiting-quota"
    assert workers.paid_fallback_attempted is False
