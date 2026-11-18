"""handle_instant_text entry + progress.md / audit.jsonl surface."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.instant import handle_instant_text
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeWorkerResult


def test_handle_instant_text_accepts_chatgpt_underscore_escapes(
    tmp_path: Path,
) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    run_dir = tmp_path / "run"
    text = (
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"test\\_x.py","acceptance":"green"}]}'
    )
    result = handle_instant_text(
        text,
        project="demo",
        store=store,
        workers=workers,
        run_dir=run_dir,
    )
    assert result.ok is True
    assert result.kind == "PLAN"
    issue = store.get_issue("demo", 1)
    assert issue is not None
    assert issue.title == "test_x.py"
    progress_path = run_dir / "progress.md"
    audit_path = run_dir / "audit.jsonl"
    assert progress_path.is_file()
    assert audit_path.is_file()
    body = progress_path.read_text(encoding="utf-8")
    assert "0%" in body or "percent" in body.lower() or "Progress" in body


def test_handle_instant_text_rejects_foreign_project(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    result = handle_instant_text(
        '{"intent_id":"p1","kind":"PLAN","project":"other","title":"X",'
        '"tasks":[{"title":"T1","acceptance":"green"}]}',
        project="demo",
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    assert result.ok is False
    assert "project" in (result.error or "").lower()


def test_progress_written_after_assign(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    workers.script_assign(1, FakeWorkerResult(ok=True, commit_sha="abc", message="done"))
    run_dir = tmp_path / "run"
    handle_instant_text(
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        '"tasks":[{"title":"T1","acceptance":"green"},'
        '{"title":"T2","acceptance":"green"},'
        '{"title":"T3","acceptance":"green"}]}',
        project="demo",
        store=store,
        workers=workers,
        run_dir=run_dir,
    )
    handle_instant_text(
        '{"intent_id":"a1","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude"}',
        project="demo",
        store=store,
        workers=workers,
        run_dir=run_dir,
    )
    body = (run_dir / "progress.md").read_text(encoding="utf-8")
    assert "in-progress" in body.lower() or "active" in body.lower() or "1" in body
    assert (run_dir / "audit.jsonl").is_file()
