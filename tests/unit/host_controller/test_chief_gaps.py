"""RED→GREEN coverage for Instant chief-of-staff remaining gaps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.host_controller.chat_bridge import (
    active_project,
    clear_active_project,
    handle_project_user_turn,
    project_prompt_addon,
    route_reply,
    set_active_project,
)
from jarvis.host_controller.controller import HostController
from jarvis.host_controller.instant import handle_instant_text
from jarvis.host_controller.intent import IntentParseError, parse_intent
from jarvis.host_controller.workers import build_live_host_workers
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers, FakeWorkerResult


def _plan(project: str = "Feature X", tasks: list[dict] | None = None) -> str:
    body = {
        "kind": "PLAN",
        "intent_id": "p-1",
        "project": project,
        "title": project,
        "tasks": tasks
        or [
            {"title": "Model", "acceptance": "tests pass"},
            {"title": "API", "acceptance": "endpoint green", "depends_on": [1]},
        ],
    }
    return json.dumps(body)


def test_production_workers_register_live_drivers(tmp_path: Path) -> None:
    workers = build_live_host_workers(repo_root=tmp_path, work_root=tmp_path / "work")
    assert "claude" in workers._drivers
    assert "grok" in workers._drivers
    assert "codex_fix" in workers._drivers
    assert "codex_review" in workers._drivers


def test_route_reply_uses_live_drivers_when_workers_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production path must not construct a bare HostWorkerBundle."""
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X")
    store.create_issue("Feature X", "Model", "ok")

    seen: dict[str, object] = {}

    def _fake_build(*, repo_root, work_root, recorded_sha_lookup=None):  # noqa: ANN001
        workers = FakeHostWorkers()
        workers.script_assign(1, FakeWorkerResult(ok=True, commit_sha="abc123"))
        seen["built"] = True
        seen["work_root"] = work_root
        return workers

    monkeypatch.setattr(
        "jarvis.host_controller.workers.build_live_host_workers",
        _fake_build,
    )
    # Also patch the name chat_bridge imports at call time.
    monkeypatch.setattr(
        "jarvis.host_controller.chat_bridge.build_live_host_workers",
        _fake_build,
        raising=False,
    )

    assign = (
        '{"kind":"ASSIGN_TASK","intent_id":"a1","project":"Feature X",'
        '"issue_number":1,"worker":"claude","task":"do it",'
        '"allow_subagents":false,"subagent_kind":""}'
    )
    # Import after patch target is set on the module chat_bridge will use.
    from jarvis.host_controller import chat_bridge as cb

    monkeypatch.setattr(cb, "build_live_host_workers", _fake_build, raising=False)

    out = route_reply(assign, data_dir=tmp_path, store=store, workers=None)
    assert seen.get("built") is True
    assert out is not None
    assert "Nicht ausgeführt" not in out  # i18n-allow


def test_state_package_is_bounded_and_includes_git_and_issues(tmp_path: Path) -> None:
    from jarvis.host_controller.state_package import build_state_package

    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X", description="ship it")
    store.create_issue("Feature X", "Model", "tests pass")
    (tmp_path / "progress.md").write_text("12.5% done\n", encoding="utf-8")

    text = build_state_package(
        project="Feature X",
        store=store,
        repo_root=Path("."),
        progress_path=tmp_path / "progress.md",
        open_prs=[{"number": 9, "title": "feat: model", "head": "feat/model"}],
        git_branch="grok/chief-of-staff",
        git_log=["eaab84ed3 feat(chat): project mode"],
        max_bytes=8192,
    )
    assert "grok/chief-of-staff" in text
    assert "Model" in text
    assert "PR #9" in text or "#9" in text
    assert "12.5%" in text
    assert len(text.encode("utf-8")) <= 8192


def test_status_path_answers_from_github_not_model(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X")
    store.create_issue("Feature X", "Model", "ok", status="in-progress")
    out = handle_project_user_turn(
        "STATUS",
        data_dir=tmp_path,
        store=store,
        workers=FakeHostWorkers(),
        run_dir=tmp_path / "run",
    )
    assert out is not None
    assert "Model" in out or "in-progress" in out or "1" in out


def test_memory_bridge_injects_top_snippets_and_remembers(tmp_path: Path) -> None:
    from jarvis.host_controller.memory_bridge import MemoryBridge

    searches: list[str] = []
    remembers: list[str] = []

    def search(query: str) -> list[dict]:
        searches.append(query)
        return [
            {"title": "Note A", "preview": "Use worktrees for agents."},
            {"title": "Note B", "preview": "Never commit secrets."},
        ]

    def remember(text: str) -> None:
        remembers.append(text)

    bridge = MemoryBridge(searcher=search, remember_fn=remember, max_snippets=2)
    block = bridge.prompt_snippets("Feature X host controller")
    assert "Note A" in block and "worktrees" in block
    assert len(block.encode("utf-8")) < 4000
    bridge.remember_decision("Accepted issue #1 with sha deadbeef")
    assert remembers and "Accepted" in remembers[0]
    assert "transcript" not in remembers[0].lower()


def test_risky_intent_is_parked_until_ja(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X")
    issue = store.create_issue("Feature X", "Cleanup", "delete old data")
    workers = FakeHostWorkers()
    workers.script_assign(issue.number, FakeWorkerResult(ok=True, commit_sha="should-not-run"))

    risky = json.dumps(
        {
            "kind": "ASSIGN_TASK",
            "intent_id": "risk-1",
            "project": "Feature X",
            "issue_number": issue.number,
            "worker": "claude",
            "task": "merge into main and delete the old data directory",
            "allow_subagents": False,
            "subagent_kind": "",
        }
    )
    out = route_reply(
        risky,
        data_dir=tmp_path,
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    assert out is not None
    assert "?" in out or "ja" in out.lower() or "nein" in out.lower()
    assert workers.calls == []
    state = active_project(data_dir=tmp_path)
    assert state is not None
    assert state.get("pending_action")

    denied = handle_project_user_turn(
        "nein",
        data_dir=tmp_path,
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    assert denied is not None
    assert workers.calls == []
    assert active_project(data_dir=tmp_path).get("pending_action") in (None, {})

    # Re-park and approve with ja.
    route_reply(
        risky,
        data_dir=tmp_path,
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    approved = handle_project_user_turn(
        "ja",
        data_dir=tmp_path,
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    assert approved is not None
    assert any(call.kind == "assign" for call in workers.calls)


def test_depends_on_blocks_assign_until_accepted(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    run_dir = tmp_path / "run"
    handle_instant_text(
        _plan(),
        "Feature X",
        store=store,
        workers=workers,
        run_dir=run_dir,
    )
    issues = store.list_issues("Feature X")
    assert len(issues) == 2
    blocked = issues[1]
    assert getattr(blocked, "depends_on", None) in ([1], [issues[0].number])

    assign = json.dumps(
        {
            "kind": "ASSIGN_TASK",
            "intent_id": "a-dep",
            "project": "Feature X",
            "issue_number": blocked.number,
            "worker": "claude",
            "task": "build API",
            "allow_subagents": False,
            "subagent_kind": "",
        }
    )
    result = handle_instant_text(assign, "Feature X", store=store, workers=workers, run_dir=run_dir)
    assert result.ok is False
    assert "depend" in (result.error or "").lower()


def test_file_overlap_refuses_second_assign(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    run_dir = tmp_path / "run"
    store.create_milestone("Feature X")
    a = store.create_issue("Feature X", "A", "ok")
    b = store.create_issue("Feature X", "B", "ok")
    store.set_status("Feature X", a.number, "in-progress")
    store.set_files("Feature X", a.number, ["jarvis/host_controller/controller.py"])

    assign = json.dumps(
        {
            "kind": "ASSIGN_TASK",
            "intent_id": "a-overlap",
            "project": "Feature X",
            "issue_number": b.number,
            "worker": "grok",
            "task": "edit same file",
            "allow_subagents": False,
            "subagent_kind": "",
            "files": ["jarvis/host_controller/controller.py"],
        }
    )
    result = handle_instant_text(assign, "Feature X", store=store, workers=workers, run_dir=run_dir)
    assert result.ok is False
    assert "overlap" in (result.error or "").lower() or "file" in (result.error or "").lower()


def test_progress_lists_ready_tasks(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    handle_instant_text(
        _plan(),
        "Feature X",
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    result = handle_instant_text(
        '{"kind":"REPORT_PROGRESS","intent_id":"r1","project":"Feature X"}',
        "Feature X",
        store=store,
        workers=workers,
        run_dir=tmp_path / "run",
    )
    assert result.ok
    ready = result.data.get("ready") or []
    assert 1 in ready or any(item == 1 or item.get("issue_number") == 1 for item in ready)


def test_research_before_plan_writes_milestone_description(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()

    def search_code(query: str) -> list[dict]:
        return [{"path": "jarvis/foo.py", "text": "feature x helper"}]

    def list_prs(query: str) -> list[dict]:
        return [{"number": 42, "title": "Feature X draft"}]

    host = HostController(
        store=store,
        workers=workers,
        audit_path=tmp_path / "audit.jsonl",
        progress_path=tmp_path / "progress.md",
        code_search=search_code,
        pr_search=list_prs,
    )
    result = host.handle(_plan(tasks=[{"title": "Ship Feature X", "acceptance": "done"}]))
    assert result.ok
    ms = store.get_milestone("Feature X")
    assert ms is not None
    assert "Feature X draft" in ms.description or "#42" in ms.description
    assert "jarvis/foo.py" in ms.description
    assert result.data.get("research")


def test_final_report_written_at_100_percent(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    workers = FakeHostWorkers()
    run_dir = tmp_path / "run"
    store.create_milestone("Feature X")
    issue = store.create_issue("Feature X", "Only", "ok")
    store.set_commit_sha("Feature X", issue.number, "deadbeef")
    store.set_review_pass_sha("Feature X", issue.number, "deadbeef")
    store.set_status("Feature X", issue.number, "review")

    host = HostController(
        store=store,
        workers=workers,
        audit_path=run_dir / "audit.jsonl",
        progress_path=run_dir / "progress.md",
        report_path=run_dir / "report.md",
    )
    # Seed a gate pass via a prior review record.
    host.gates.record_review(
        "Feature X",
        issue.number,
        commit_sha="deadbeef",
        pass_verdict=True,
        findings=[],
    )
    accept = json.dumps(
        {
            "kind": "ACCEPT_TASK",
            "intent_id": "acc",
            "project": "Feature X",
            "issue_number": issue.number,
            "commit_sha": "deadbeef",
            "tests_green": True,
        }
    )
    result = host.handle(accept)
    assert result.ok
    report = run_dir / "report.md"
    assert report.is_file()
    body = report.read_text(encoding="utf-8")
    assert "deadbeef" in body
    assert "100" in body or "accepted" in body.lower()


def test_normal_chat_prompt_unchanged_when_project_mode_off(tmp_path: Path) -> None:
    clear_active_project(data_dir=tmp_path)
    base = "You are Dodaios."
    addon = project_prompt_addon(data_dir=tmp_path, store=None, repo_root=tmp_path)
    assert addon == ""
    assert base + addon == base
    # Ordinary message with mode off takes the normal path.
    assert (
        route_reply(
            "Hallo", data_dir=tmp_path, store=FakeGitHubProjectStore(), workers=FakeHostWorkers()
        )
        is None
    )


def test_plan_task_rejects_unknown_keys_still() -> None:
    with pytest.raises(IntentParseError):
        parse_intent(
            json.dumps(
                {
                    "kind": "PLAN",
                    "intent_id": "x",
                    "project": "P",
                    "title": "P",
                    "tasks": [{"title": "t", "acceptance": "a", "hack": 1}],
                }
            )
        )


@pytest.mark.skipif(
    __import__("os").environ.get("DODAIOS_HOST_LIVE") != "1",
    reason="live GitHub/Obsidian require DODAIOS_HOST_LIVE=1",
)
def test_live_research_and_memory_roundtrip() -> None:
    from jarvis.host_controller.memory_bridge import MemoryBridge
    from jarvis.host_controller.research import default_code_search, default_pr_search

    code = default_code_search("host_controller", repo="Qualle1980/Dodaios")
    prs = default_pr_search("host controller", repo="Qualle1980/Dodaios")
    assert isinstance(code, list)
    assert isinstance(prs, list)
    bridge = MemoryBridge()
    snippets = bridge.prompt_snippets("Dodaios host controller")
    assert isinstance(snippets, str)
    # remember is side-effecting; only call under live flag with a tiny note.
    bridge.remember_decision("Live host-controller memory bridge smoke (no secrets).")
