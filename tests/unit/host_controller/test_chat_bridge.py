"""Project mode: state file, host intent from a chat reply, and the prompt directive."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.host_controller.chat_bridge import (
    active_project,
    clear_active_project,
    instant_directive,
    parse_host_intent,
    route_reply,
    set_active_project,
)
from tests.fakes.fake_host_github import FakeGitHubProjectStore
from tests.fakes.fake_host_workers import FakeHostWorkers

PLAN = (
    '{"kind": "PLAN", "intent_id": "p-1", "project": "Feature X", '
    '"title": "Feature X", "tasks": [{"title": "Datenmodell", "acceptance": "tests pass"}]}'
)


def _store() -> FakeGitHubProjectStore:
    return FakeGitHubProjectStore()


def test_project_mode_is_off_by_default(tmp_path: Path) -> None:
    assert active_project(data_dir=tmp_path) is None
    assert route_reply(PLAN, data_dir=tmp_path, store=_store(), workers=FakeHostWorkers()) is None


def test_set_read_and_clear_the_active_project(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    assert active_project(data_dir=tmp_path) == {
        "project": "Feature X",
        "repo": "Qualle1980/Dodaios",
    }
    assert clear_active_project(data_dir=tmp_path) is True
    assert active_project(data_dir=tmp_path) is None
    assert clear_active_project(data_dir=tmp_path) is False


def test_set_rejects_a_repo_without_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        set_active_project("Feature X", "Dodaios", data_dir=tmp_path)


def test_a_broken_state_file_means_project_mode_is_off(tmp_path: Path) -> None:
    (tmp_path / "host_active_project.json").write_text("{not json", encoding="utf-8")
    assert active_project(data_dir=tmp_path) is None


def test_plain_text_takes_the_normal_path(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    out = route_reply(
        "Django ist ein Webframework.",  # i18n-allow
        data_dir=tmp_path,
        store=_store(),
        workers=FakeHostWorkers(),  # i18n-allow # noqa: E501
    )
    assert out is None


def test_route_envelope_is_not_a_host_intent(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    out = route_reply(
        '{"route": "clarify", "text": "Was genau?"}',  # i18n-allow # noqa: E501
        data_dir=tmp_path,
        store=_store(),
        workers=FakeHostWorkers(),
    )
    assert out is None


def test_unknown_intent_kind_is_reported_as_rejected(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    out = route_reply(
        '{"kind": "DELETE_ALL", "intent_id": "x"}',
        data_dir=tmp_path,
        store=_store(),
        workers=FakeHostWorkers(),
    )
    assert out is not None and out.startswith(
        "Nicht ausgeführt"  # i18n-allow
    )  # i18n-allow # noqa: E501


def test_a_plan_intent_is_executed_and_summarised(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    store = _store()
    out = route_reply(PLAN, data_dir=tmp_path, store=store, workers=FakeHostWorkers())
    assert (
        out == "Plan angelegt: 1 Aufgabe(n) im Projekt Feature X."
    )  # i18n-allow # noqa: E501
    assert store.get_milestone("Feature X") is not None


def test_a_fenced_plan_is_accepted(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    out = route_reply(
        f"```json\n{PLAN}\n```", data_dir=tmp_path, store=_store(), workers=FakeHostWorkers()
    )
    assert out is not None and out.startswith(
        "Plan angelegt"
    )  # i18n-allow # noqa: E501


def test_an_escaped_underscore_still_executes(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    bs = chr(92)
    escaped = PLAN.replace('"Feature X", "tasks"', '"Feature X ' + bs + '_Y", "tasks"').replace(
        '"Datenmodell"', '"Daten' + bs + '_modell"'
    )
    out = route_reply(escaped, data_dir=tmp_path, store=_store(), workers=FakeHostWorkers())
    assert out is not None and out.startswith(
        "Plan angelegt"
    )  # i18n-allow # noqa: E501


def test_a_rejected_intent_is_reported_not_hidden(tmp_path: Path) -> None:
    set_active_project("Feature X", "Qualle1980/Dodaios", data_dir=tmp_path)
    bad = PLAN.replace('"Datenmodell"', '"Daten' + chr(92) + 'xmodell"')
    out = route_reply(bad, data_dir=tmp_path, store=_store(), workers=FakeHostWorkers())
    assert out is not None and out.startswith(
        "Nicht ausgeführt"  # i18n-allow
    )  # i18n-allow # noqa: E501


def test_parse_host_intent_returns_none_for_prose() -> None:
    assert parse_host_intent("Hallo, wie geht es?") is None
    assert parse_host_intent("[1, 2]") is None
    assert parse_host_intent('{"route": "self", "text": "x"}') is None


def test_directive_is_the_instant_skill_with_every_intent_kind() -> None:
    text = instant_directive()
    for kind in (
        "PLAN",
        "ASSIGN_TASK",
        "REQUEST_REVIEW",
        "REQUEST_FIX",
        "ACCEPT_TASK",
        "BLOCKED",
        "REPORT_PROGRESS",
    ):
        assert kind in text
