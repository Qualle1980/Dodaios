# ruff: noqa: E501 (German runtime phrases under test are long)
"""End-state honesty: a task is finished only when the host accepted it."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.host_controller.chat_bridge import (
    clear_active_project,
    route_reply,
    set_active_project,
)
from jarvis.host_controller.controller import HostController, HostResult
from jarvis.host_controller.end_state import (
    claims_finished,
    finished_user_phrase,
    honest_completion_reply,
    running_user_phrase,
    user_end_state_phrase,
)
from jarvis.host_controller.instant import build_user_reply, scrub_finished_claim
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers


def _store_with_open_work() -> FakeGitHubProjectStore:
    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X")
    a = store.create_issue("Feature X", "One", "ok")
    b = store.create_issue("Feature X", "Two", "ok")
    store.set_status("Feature X", a.number, "accepted")
    store.set_commit_sha("Feature X", a.number, "aaa")
    store.set_status("Feature X", b.number, "in-progress")
    store.set_worker("Feature X", b.number, "claude")
    return store


def _store_all_accepted() -> FakeGitHubProjectStore:
    store = FakeGitHubProjectStore()
    store.create_milestone("Feature X")
    a = store.create_issue("Feature X", "One", "ok")
    b = store.create_issue("Feature X", "Two", "ok")
    for issue in (a, b):
        store.set_status("Feature X", issue.number, "accepted")
        store.set_commit_sha("Feature X", issue.number, "deadbeef")
    return store


def test_claims_finished_matches_de_and_en() -> None:
    assert claims_finished(
        "Das Projekt ist fertig."  # i18n-allow: German runtime phrase under test
    )  # i18n-allow: German runtime phrase under test
    assert claims_finished("Auftrag erledigt!")
    assert claims_finished("Alles abgeschlossen.")
    assert claims_finished("The project is done.")
    assert claims_finished("All tasks finished.")
    assert not claims_finished("Stand: 50 % abgenommen, 1 in Arbeit.")
    assert not claims_finished("läuft noch, 50 %")  # i18n-allow: German runtime phrase under test


def test_running_phrase_carries_percent() -> None:
    assert (
        running_user_phrase(50) == "läuft noch, 50 %"  # i18n-allow: German runtime phrase under test
    )  # i18n-allow: German runtime phrase under test
    assert (
        running_user_phrase(0.0) == "läuft noch, 0 %"  # i18n-allow: German runtime phrase under test
    )  # i18n-allow: German runtime phrase under test
    assert "still running" in running_user_phrase(25, language="en")


def test_user_end_state_says_in_progress_while_open() -> (
    None
):  # i18n-allow: German runtime phrase under test
    store = _store_with_open_work()
    progress = store.progress("Feature X")
    issues = list(store.list_issues("Feature X"))
    phrase = user_end_state_phrase(progress, issues)
    assert phrase.startswith("läuft noch,")  # i18n-allow: German runtime phrase under test
    assert "%" in phrase
    assert "fertig" not in phrase  # i18n-allow: German runtime phrase under test
    assert progress["percent"] == 50.0


def test_user_end_state_fertig_only_when_all_accepted() -> (  # i18n-allow: German runtime phrase under test
    None
):  # i18n-allow: German runtime phrase under test
    store = _store_all_accepted()
    progress = store.progress("Feature X")
    issues = list(store.list_issues("Feature X"))
    assert user_end_state_phrase(progress, issues) == finished_user_phrase()
    assert progress["percent"] == 100.0


def test_honest_completion_rewrites_fertig_while_open() -> (  # i18n-allow: German runtime phrase under test
    None
):  # i18n-allow: German runtime phrase under test
    store = _store_with_open_work()
    progress = store.progress("Feature X")
    issues = list(store.list_issues("Feature X"))
    out = honest_completion_reply(
        "Alles fertig, Auftrag erledigt.",  # i18n-allow: German runtime phrase under test
        progress=progress,
        issues=issues,
    )
    assert out == "läuft noch, 50 %"  # i18n-allow: German runtime phrase under test


def test_honest_completion_keeps_fertig_when_accepted() -> (  # i18n-allow: German runtime phrase under test
    None
):  # i18n-allow: German runtime phrase under test
    store = _store_all_accepted()
    progress = store.progress("Feature X")
    issues = list(store.list_issues("Feature X"))
    text = "Alles fertig."  # i18n-allow: German runtime phrase under test
    assert honest_completion_reply(text, progress=progress, issues=issues) == text


def test_build_user_reply_reports_laeuft_noch(  # i18n-allow: German runtime phrase under test
    tmp_path: Path,
) -> None:  # i18n-allow: German runtime phrase under test
    store = _store_with_open_work()
    result = HostResult(
        ok=True,
        kind="REPORT_PROGRESS",
        data=dict(store.progress("Feature X")),
    )
    reply = build_user_reply(result, store=store, project="Feature X")
    assert reply.startswith("läuft noch, 50 %")  # i18n-allow: German runtime phrase under test
    assert "fertig" not in reply.casefold()  # i18n-allow: German runtime phrase under test


def test_build_user_reply_fertig_after_final_accept(  # i18n-allow: German runtime phrase under test
    tmp_path: Path,
) -> None:  # i18n-allow: German runtime phrase under test
    store = _store_all_accepted()
    result = HostResult(
        ok=True,
        kind="ACCEPT_TASK",
        data={"final_report": True, "issue_number": 2, "status": "accepted"},
    )
    reply = build_user_reply(result, store=store, project="Feature X")
    assert reply.startswith("fertig")  # i18n-allow: German runtime phrase under test
    assert "Abschlussbericht" in reply


def test_scrub_and_route_reply_block_false_fertig(  # i18n-allow: German runtime phrase under test
    tmp_path: Path,
) -> None:  # i18n-allow: German runtime phrase under test
    store = _store_with_open_work()
    set_active_project("Feature X", "owner/repo", data_dir=tmp_path)
    try:
        scrubbed = scrub_finished_claim(
            "Der Auftrag ist fertig.",  # i18n-allow: German runtime phrase under test
            store=store,
            project="Feature X",
        )
        assert scrubbed == "läuft noch, 50 %"  # i18n-allow: German runtime phrase under test

        routed = route_reply(
            "Alles erledigt, wir sind done.",
            data_dir=tmp_path,
            store=store,
            workers=FakeHostWorkers(),
            run_dir=tmp_path / "run",
        )
        assert routed == "läuft noch, 50 %"  # i18n-allow: German runtime phrase under test
    finally:
        clear_active_project(data_dir=tmp_path)


def test_route_reply_allows_fertig_when_host_finished(  # i18n-allow: German runtime phrase under test
    tmp_path: Path,
) -> None:  # i18n-allow: German runtime phrase under test
    store = _store_all_accepted()
    set_active_project("Feature X", "owner/repo", data_dir=tmp_path)
    try:
        routed = route_reply(
            "Alles fertig.",  # i18n-allow: German runtime phrase under test
            data_dir=tmp_path,
            store=store,
            workers=FakeHostWorkers(),
            run_dir=tmp_path / "run",
        )
        # Host agrees — Instant prose stays (route_reply returns None).
        assert routed is None
    finally:
        clear_active_project(data_dir=tmp_path)


def test_progress_md_says_laeuft_noch_while_open(  # i18n-allow: German runtime phrase under test
    tmp_path: Path,
) -> None:  # i18n-allow: German runtime phrase under test
    store = _store_with_open_work()
    workers = FakeHostWorkers()
    host = HostController(
        store=store,
        workers=workers,
        audit_path=tmp_path / "audit.jsonl",
        progress_path=tmp_path / "progress.md",
        progress_title="end-state",
    )
    intent = json.dumps(
        {
            "kind": "REPORT_PROGRESS",
            "intent_id": "p1",
            "project": "Feature X",
        }
    )
    from jarvis.host_controller.instant import handle_instant_text

    result = handle_instant_text(
        intent,
        "Feature X",
        host=host,
        run_dir=tmp_path,
    )
    assert result.ok
    body = (tmp_path / "progress.md").read_text(encoding="utf-8")
    assert "läuft noch, 50 %" in body  # i18n-allow: German runtime phrase under test
    assert "project complete" not in body
