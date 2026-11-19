"""Progress counts only accepted tasks from GitHub, never agent claims."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.controller import HostController
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers


def test_progress_zero_when_none_accepted(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    host = HostController(
        store=store,
        workers=workers,
        audit_path=tmp_path / "audit.jsonl",
        max_runs_per_project=50,
    )
    tasks = ",".join(
        f'{{"title":"T{i}","acceptance":"green"}}' for i in range(1, 11)
    )
    host.handle(
        f'{{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        f'"tasks":[{tasks}]}}'
    )
    for number in (1, 2, 3):
        store.set_status("demo", number, "in-progress")

    report = host.handle(
        '{"intent_id":"rp1","kind":"REPORT_PROGRESS","project":"demo"}'
    )
    assert report.ok is True
    assert report.data["planned"] == 10
    assert report.data["accepted"] == 0
    assert report.data["in_progress"] == 3
    assert report.data["percent"] == 0.0
