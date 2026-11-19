"""PLAN may name earlier tasks by index; the host resolves them to issue numbers."""

from __future__ import annotations

from pathlib import Path

from jarvis.host_controller.instant import handle_instant_text
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers


def _plan(store: FakeGitHubProjectStore, tmp_path: Path, tasks_json: str):
    text = (
        '{"intent_id":"p1","kind":"PLAN","project":"demo","title":"Demo",'
        f'"tasks":[{tasks_json}]}}'
    )
    return handle_instant_text(
        text, project="demo", store=store, workers=FakeHostWorkers(), run_dir=tmp_path
    )


def test_dependency_by_index_resolves_to_the_earlier_issue(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    result = _plan(
        store,
        tmp_path,
        '{"title":"A","acceptance":"tests green"},'
        '{"title":"B","acceptance":"endpoint answers","depends_on_tasks":[0]}',
    )
    assert result.ok, result.error
    a, b = result.data["issues"]
    assert store.get_issue("demo", b).depends_on == [a]
    assert store.get_issue("demo", a).depends_on in ([], None)


def test_a_forward_or_self_reference_is_rejected(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    result = _plan(
        store,
        tmp_path,
        '{"title":"A","acceptance":"x","depends_on_tasks":[1]},'
        '{"title":"B","acceptance":"y"}',
    )
    assert not result.ok
    assert "earlier task" in (result.error or "")


def test_a_bool_index_is_rejected(tmp_path: Path) -> None:
    store = FakeGitHubProjectStore()
    result = _plan(
        store,
        tmp_path,
        '{"title":"A","acceptance":"x"},'
        '{"title":"B","acceptance":"y","depends_on_tasks":[true]}',
    )
    assert not result.ok


def test_approval_question_names_the_plan_and_its_tasks() -> None:
    from jarvis.host_controller.approval import approval_question
    from jarvis.host_controller.intent import parse_intent

    parsed = parse_intent(
        '{"intent_id":"p9","kind":"PLAN","project":"demo","title":"Release",'
        '"tasks":[{"title":"Merge nach main","acceptance":"tests green"}]}'
    )
    question = approval_question(parsed)
    assert "Release" in question and "Merge nach main" in question
