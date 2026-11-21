"""A live turn's file read, executed through the real supervisor gateway.

Not a fake tool and not a source-text assertion: the gateway is the shipped
`BrainSupervisorToolGateway`, the project root comes from the real Agentic-IDE
registry, and the read runs through `gateway.execute(..., origin="realtime")`,
which is the one authorised path (AP-3). No provider, no audio, no network.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import jarvis.agentic_ide.session as session_mod
from jarvis.agentic_ide.folders import probe_project
from jarvis.agentic_ide.session import Registry, Session
from jarvis.brain.factory import _active_voice_project_root
from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.core.protocols import SupervisorToolRequest

FIRST_LINE = "the first line of the harmless test file"


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """An isolated, focused project exactly as the app would report one."""
    root = tmp_path / "voice-test-project"
    root.mkdir()
    (root / "note.txt").write_text(f"{FIRST_LINE}\nsecond line\n", encoding="utf-8")

    live = Registry()
    workspace = Session(
        id="ws-voice",
        folder=str(root),
        name=root.name,
        profile=probe_project(root),
        terminals=[],
        created_at=time.time(),
        focus_mode=True,
    )
    live._sessions[workspace.id] = workspace
    live._active = workspace.id
    monkeypatch.setattr(session_mod, "_REGISTRY", live)
    return root


def _gateway() -> BrainSupervisorToolGateway:
    """The shipped gateway over the shipped executor -- risk tiers and all."""
    from jarvis.core.bus import EventBus
    from jarvis.core.config import JarvisConfig
    from jarvis.safety import ApprovalWorkflow, RiskTierEvaluator, ToolExecutor

    bus = EventBus()
    executor = ToolExecutor(
        bus, RiskTierEvaluator(JarvisConfig().safety), ApprovalWorkflow(bus)
    )
    manager = SimpleNamespace(
        _tools={},
        _tool_executor=executor,
        _config=SimpleNamespace(computer_use=SimpleNamespace(enabled=False)),
    )
    return BrainSupervisorToolGateway(
        manager, voice_project_resolver=_active_voice_project_root
    )


def _request() -> SupervisorToolRequest:
    return SupervisorToolRequest(
        trace_id=uuid4(),
        origin="realtime",
        user_utterance="Read the first line of note.txt.",
    )


@pytest.mark.asyncio
async def test_a_live_turn_really_reads_a_project_file(project: Path) -> None:
    result = await _gateway().execute("Read", {"file_path": "note.txt"}, _request())

    assert result.success is True
    output = str(result.output or "")
    assert FIRST_LINE in output
    # The answer is grounded in the real file, not in the tool name.
    assert "second line" in output


@pytest.mark.asyncio
async def test_a_failed_read_reports_failure_instead_of_a_result(project: Path) -> None:
    """The model must never be handed something it can call a successful read."""
    result = await _gateway().execute("Read", {"file_path": "absent.txt"}, _request())

    assert result.success is False
    combined = f"{result.output or ''} {result.error or ''}"
    assert FIRST_LINE not in combined
    assert "not found" in combined.lower() or "no such" in combined.lower()


@pytest.mark.asyncio
async def test_a_live_turn_cannot_read_outside_the_focused_project(
    project: Path, tmp_path: Path
) -> None:
    secret = tmp_path / "outside.txt"
    secret.write_text("SUPERSECRET=42", encoding="utf-8")

    result = await _gateway().execute(
        "Read", {"file_path": "../outside.txt"}, _request()
    )

    assert result.success is False
    assert "SUPERSECRET=42" not in f"{result.output or ''} {result.error or ''}"


@pytest.mark.asyncio
async def test_the_live_turn_has_no_write_edit_or_shell_tool(project: Path) -> None:
    gateway = _gateway()
    offered = {descriptor.name for descriptor in gateway.voice_catalog()}

    assert {"Read", "Ls", "Glob", "Grep"} <= offered
    assert not {"Write", "Edit", "RunCommand"} & offered

    for forbidden in ("Write", "Edit", "RunCommand"):
        result = await gateway.execute(
            forbidden, {"file_path": "note.txt", "content": "x"}, _request()
        )
        assert result.success is False


@pytest.mark.asyncio
async def test_losing_focus_revokes_the_read_mid_conversation(project: Path) -> None:
    """The catalog is rebuilt per turn, so a dropped focus takes the tools."""
    gateway = _gateway()
    assert (await gateway.execute("Read", {"file_path": "note.txt"}, _request())).success

    session_mod.get_registry().session.focus_mode = False

    assert not {"Read", "Ls", "Glob", "Grep"} & {
        descriptor.name for descriptor in gateway.voice_catalog()
    }
    revoked = await gateway.execute("Read", {"file_path": "note.txt"}, _request())
    assert revoked.success is False
    assert FIRST_LINE not in f"{revoked.output or ''} {revoked.error or ''}"
