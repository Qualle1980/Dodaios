"""Regression: chat-addressed Claude panes must never silently pick Opus."""

from __future__ import annotations

import pytest

from jarvis.agentic_ide.orchestration import (
    _chat_pane_model,
    _chat_pane_permission,
)
from jarvis.agentic_ide.session import SessionError
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool
from jarvis.workspace.launch_picks import launch_argv, normalize_model


def test_chat_claude_defaults_to_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", raising=False)
    assert _chat_pane_model("claude") == "sonnet"
    assert _chat_pane_permission("claude") == "acceptEdits"


def test_chat_claude_can_explicitly_use_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", raising=False)
    assert _chat_pane_model("claude", "haiku") == "haiku"


@pytest.mark.parametrize("requested", ["opus", "claude-opus-5", "default", "unknown"])
def test_chat_claude_refuses_unapproved_model(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", raising=False)
    with pytest.raises(SessionError, match="sonnet or haiku"):
        _chat_pane_model("claude", requested)


def test_chat_claude_refuses_inherited_scrubbing_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "1")
    with pytest.raises(SessionError, match="ordinary desktop shortcut"):
        _chat_pane_model("claude")


def test_chat_claude_does_not_turn_off_security_scrubbing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "1")
    with pytest.raises(SessionError):
        _chat_pane_model("claude")
    assert __import__("os").environ["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"


def test_argv_pins_model_and_uses_approve_edits_not_bypass() -> None:
    assert normalize_model("claude", "sonnet") == "sonnet"
    argv = launch_argv("claude", model="sonnet", permission_mode="acceptEdits")
    assert ("--model", "sonnet") == argv[:2]
    assert "--permission-mode" in argv
    assert "acceptEdits" in argv
    assert "--dangerously-skip-permissions" not in argv


def test_other_cli_unchanged() -> None:
    assert _chat_pane_model("codex") == ""
    assert _chat_pane_permission("codex") is None


def test_chat_tool_exposes_only_approved_claude_model_tiers() -> None:
    props = WorkspaceOrchestrationTool.schema["properties"]
    assert props["model"]["enum"] == ["sonnet", "haiku"]
