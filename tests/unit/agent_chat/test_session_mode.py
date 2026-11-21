"""Session work mode (``project`` | ``chat``): store, migration, cwd rules."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import get_args

import pytest
from fastapi import HTTPException

from jarvis.agent_chat import store as store_mod
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import (
    DEFAULT_SESSION_MODE,
    SESSION_MODES,
    AgentChatStore,
)
from jarvis.ui.web import agent_chat_routes


def test_python_and_pydantic_spell_the_mode_the_same():
    assert set(get_args(agent_chat_routes.SessionModeName)) == set(SESSION_MODES)
    body_field = agent_chat_routes.CreateSessionBody.model_fields["mode"]
    assert set(get_args(body_field.annotation)) == set(SESSION_MODES)
    assert body_field.default == DEFAULT_SESSION_MODE


def test_old_rows_read_as_chat_mode(tmp_path: Path):
    """A database from before the column: every row becomes chat mode."""
    db = tmp_path / "agent_chat.db"
    column = ",\n    mode             TEXT NOT NULL DEFAULT 'chat'"
    old_schema = store_mod._SCHEMA.replace(column, "")
    assert "mode             TEXT" not in old_schema, "the old schema must not know the column"
    conn = sqlite3.connect(db)
    conn.executescript(old_schema)
    conn.execute(
        "INSERT INTO agent_chat_sessions (session_id, provider, created_ms, updated_ms) "
        "VALUES ('old1', 'openai', 1, 1)"
    )
    conn.commit()
    conn.close()

    store = AgentChatStore(db)
    old = store.get_session("old1")
    assert old is not None and old.mode == "chat"
    store.close()
    again = AgentChatStore(db)
    assert again.get_session("old1") is not None
    again.close()


def test_create_session_defaults_to_chat_without_folder():
    store = AgentChatStore(":memory:")
    s = store.create_session(provider="openai", model="m", effort="", cwd="")
    assert s.mode == "chat"
    assert s.to_dict()["mode"] == "chat"


def test_create_session_stores_project_mode(tmp_path: Path):
    store = AgentChatStore(":memory:")
    s = store.create_session(
        provider="openai", model="m", effort="", cwd=str(tmp_path), mode="project"
    )
    assert s.mode == "project"
    assert s.cwd == str(tmp_path)


def test_unknown_mode_is_refused():
    store = AgentChatStore(":memory:")
    with pytest.raises(ValueError, match="mode must be one of"):
        store.create_session(provider="openai", model="m", effort="", cwd=".", mode="voice")


def test_update_session_can_set_mode(tmp_path: Path):
    store = AgentChatStore(":memory:")
    s = store.create_session(provider="openai", model="m", effort="", cwd="")
    store.update_session(s.session_id, mode="project", cwd=str(tmp_path))
    after = store.get_session(s.session_id)
    assert after is not None and after.mode == "project" and after.cwd == str(tmp_path)


def test_service_chat_mode_keeps_optional_cwd_without_becoming_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Chat may hold a preferred folder; mode stays chat (not project)."""
    from jarvis.agent_chat import runner_api

    monkeypatch.setitem(
        runner_api.BRAIN_BY_PROVIDER,
        "fakeprov",
        ("tests.unit.agent_chat.test_agent_chat_service", "ScriptedBrain"),
    )
    svc = AgentChatService(store=AgentChatStore(":memory:"))
    session = svc.create_session(provider="fakeprov", cwd=str(tmp_path), mode="chat")
    assert session.mode == "chat"
    assert session.cwd == str(tmp_path)

    empty = svc.create_session(provider="fakeprov", cwd="", mode="chat")
    assert empty.mode == "chat"
    assert empty.cwd == ""


def test_service_project_mode_requires_cwd(monkeypatch: pytest.MonkeyPatch):
    from jarvis.agent_chat import runner_api

    monkeypatch.setitem(
        runner_api.BRAIN_BY_PROVIDER,
        "fakeprov",
        ("tests.unit.agent_chat.test_agent_chat_service", "ScriptedBrain"),
    )
    svc = AgentChatService(store=AgentChatStore(":memory:"))
    with pytest.raises(ValueError, match="project mode requires a working folder"):
        svc.create_session(provider="fakeprov", mode="project")


def test_chat_mode_with_cwd_still_gates_writes(tmp_path: Path):
    """Optional chat cwd does not turn the session into an auto-write project."""
    from jarvis.agent_chat import write_gate as wg

    main = tmp_path / "main"
    main.mkdir()
    (main / ".git").mkdir()
    d = wg.evaluate_write_gate(
        tool_name="RunCommand",
        args={"command": "git push origin HEAD"},
        mode="chat",
        cwd=str(main),
    )
    assert d.is_write and d.needs_ok
    assert "chat mode" in d.reason


def test_service_project_mode_keeps_explicit_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from jarvis.agent_chat import runner_api

    monkeypatch.setitem(
        runner_api.BRAIN_BY_PROVIDER,
        "fakeprov",
        ("tests.unit.agent_chat.test_agent_chat_service", "ScriptedBrain"),
    )
    svc = AgentChatService(store=AgentChatStore(":memory:"))
    session = svc.create_session(provider="fakeprov", cwd=str(tmp_path), mode="project")
    assert session.mode == "project"
    assert session.cwd == str(tmp_path)
    # No silent home fallback: cwd is exactly what was passed.
    assert session.cwd != str(Path.home())


def test_route_project_without_cwd_is_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    class _FakeSvc:
        def create_session(self, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("create_session must not run without cwd")

    monkeypatch.setattr(
        agent_chat_routes, "_service", lambda _request: _FakeSvc()
    )
    monkeypatch.setattr(
        agent_chat_routes,
        "resolve_runner",
        lambda *a, **k: "api",
    )
    monkeypatch.setattr(
        agent_chat_routes,
        "ladder_key",
        lambda *a, **k: "api",
    )
    monkeypatch.setattr(
        agent_chat_routes,
        "normalize_permission",
        lambda *a, **k: "auto",
    )
    monkeypatch.setattr(
        agent_chat_routes,
        "is_permission_mode",
        lambda *a, **k: True,
    )
    body = agent_chat_routes.CreateSessionBody(provider="openai", mode="project")
    with pytest.raises(HTTPException) as exc:
        agent_chat_routes.create_session(body, request=SimpleNamespace())
    assert exc.value.status_code == 400
    assert "working folder" in str(exc.value.detail)


def test_route_chat_mode_creates_without_folder(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    captured: dict = {}

    class _FakeStore:
        def save_chat_selection(self, *_a, **_k):
            return None

    class _FakeSession:
        surface = "agent"
        provider = "openai"
        model = "m"
        effort = ""
        account_id = ""
        mode = "chat"
        cwd = ""

        def to_dict(self):
            return {
                "session_id": "s1",
                "mode": self.mode,
                "cwd": self.cwd,
                "surface": self.surface,
            }

    class _FakeSvc:
        store = _FakeStore()

        def create_session(self, **kwargs):
            captured.update(kwargs)
            return _FakeSession()

    monkeypatch.setattr(agent_chat_routes, "_service", lambda _request: _FakeSvc())
    monkeypatch.setattr(agent_chat_routes, "resolve_runner", lambda *a, **k: "api")
    monkeypatch.setattr(agent_chat_routes, "ladder_key", lambda *a, **k: "api")
    monkeypatch.setattr(agent_chat_routes, "normalize_permission", lambda *a, **k: "auto")
    monkeypatch.setattr(agent_chat_routes, "is_permission_mode", lambda *a, **k: True)

    body = agent_chat_routes.CreateSessionBody(provider="openai")  # default mode=chat
    out = agent_chat_routes.create_session(body, request=SimpleNamespace())
    assert captured["mode"] == "chat"
    assert captured["cwd"] == ""
    assert out["mode"] == "chat"
