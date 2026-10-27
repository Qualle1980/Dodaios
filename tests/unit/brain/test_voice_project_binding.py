"""Windows acceptance for the voice project binding: real registry, no model call.

Every case drives the real :class:`jarvis.agentic_ide.session.Registry` and real
``Session`` objects, the same state holder the app writes, and asserts what the
live tool catalog would offer. Nothing here calls a provider or spends a key.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.agentic_ide.session as session_mod
from jarvis.agentic_ide.folders import probe_project
from jarvis.agentic_ide.session import Registry, Session
from jarvis.brain.factory import _active_voice_project_root
from jarvis.brain.tool_gateway import BrainSupervisorToolGateway

READ_TOOLS = {"Read", "Ls", "Glob", "Grep"}
MUTATING_TOOLS = {"Write", "Edit", "RunCommand"}


def _session(folder: Path, *, focus: bool, ident: str = "ws1") -> Session:
    return Session(
        id=ident,
        folder=str(folder),
        name=folder.name,
        profile=probe_project(folder),
        terminals=[],
        created_at=time.time(),
        focus_mode=focus,
    )


@pytest.fixture
def registry(monkeypatch) -> Registry:
    """A real Registry, isolated from the process-wide one."""
    live = Registry()
    monkeypatch.setattr(session_mod, "_REGISTRY", live)
    return live


def _select(registry: Registry, session: Session) -> None:
    """What the app does when a workspace becomes the visible one."""
    registry._sessions[session.id] = session
    registry._active = session.id


def _voice_tools(resolver) -> dict:
    manager = SimpleNamespace(
        _tools={"sentinel_tool": object()},
        _config=SimpleNamespace(computer_use=SimpleNamespace(enabled=False)),
    )
    gateway = BrainSupervisorToolGateway(manager, voice_project_resolver=resolver)
    return gateway._voice_tools()


# ------------------------------------------------------------------ binding


def test_project_selected_and_focused_is_readable(tmp_path: Path, registry: Registry):
    project = tmp_path / "proj"
    project.mkdir()
    _select(registry, _session(project, focus=True))

    assert _active_voice_project_root() == project.resolve()
    assert READ_TOOLS <= set(_voice_tools(_active_voice_project_root))


def test_no_project_selected_reads_nothing(registry: Registry):
    """The wizard is showing: no folder, and never the process cwd."""
    assert registry.session is None
    assert _active_voice_project_root() is None

    tools = _voice_tools(_active_voice_project_root)
    assert not READ_TOOLS & set(tools)
    assert "sentinel_tool" in tools


def test_focus_lost_reads_nothing(tmp_path: Path, registry: Registry):
    project = tmp_path / "proj"
    project.mkdir()
    _select(registry, _session(project, focus=False))

    assert _active_voice_project_root() is None
    assert not READ_TOOLS & set(_voice_tools(_active_voice_project_root))


def test_switching_project_moves_the_root(tmp_path: Path, registry: Registry):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _select(registry, _session(first, focus=True, ident="ws1"))
    assert _active_voice_project_root() == first.resolve()

    _select(registry, _session(second, focus=True, ident="ws2"))
    assert _active_voice_project_root() == second.resolve()

    # The catalog is rebuilt per turn, so the old root must not linger.
    tool = _voice_tools(_active_voice_project_root)["Read"]
    assert tool._root == second.resolve()


def test_vanished_project_folder_reads_nothing(tmp_path: Path, registry: Registry):
    """A selected folder that is gone (unplugged drive, deleted) reads nothing."""
    project = tmp_path / "gone"
    project.mkdir()
    _select(registry, _session(project, focus=True))
    project.rmdir()

    assert _active_voice_project_root() is None
    assert not READ_TOOLS & set(_voice_tools(_active_voice_project_root))


def test_project_path_is_a_file_reads_nothing(tmp_path: Path, registry: Registry):
    """An invalid selection (a file, not a directory) is not a readable root."""
    bogus = tmp_path / "not-a-dir.txt"
    bogus.write_text("x", encoding="utf-8")
    _select(registry, _session(bogus, focus=True))

    assert _active_voice_project_root() is None


def test_mutating_tools_never_reach_the_voice_path(tmp_path: Path, registry: Registry):
    project = tmp_path / "proj"
    project.mkdir()
    _select(registry, _session(project, focus=True))

    tools = _voice_tools(_active_voice_project_root)
    assert READ_TOOLS <= set(tools)
    assert not MUTATING_TOOLS & set(tools)


def test_within_rejects_a_path_that_resolves_out_of_the_root(tmp_path: Path):
    """The file-level guard, provable without symlink privileges.

    The OS-level file-symlink case needs a privilege this box does not grant,
    so the guard those tests rely on is asserted directly here.
    """
    from jarvis.agent_chat.tools import _within

    project = tmp_path / "proj"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "password.txt").write_text("x", encoding="utf-8")

    assert _within(project / "own.txt", project.resolve()) is True
    assert _within(outside / "password.txt", project.resolve()) is False
    assert _within(project / ".." / "outside" / "password.txt", project.resolve()) is False
