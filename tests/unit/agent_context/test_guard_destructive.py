"""Tests for .agent-context/tools/guard-destructive.ps1 (PreToolUse hook)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

_GUARD = Path(__file__).resolve().parents[3] / ".agent-context" / "tools" / "guard-destructive.ps1"
_SHELL = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(_SHELL is None, reason="no PowerShell on this machine")


def _run(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _SHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_GUARD),
        ],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=NO_WINDOW_CREATIONFLAGS,
        check=False,
    )


def _decision(tool: str, command: str) -> str | None:
    proc = _run(json.dumps({"tool_name": tool, "tool_input": {"command": command}}))
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return None
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    return out["permissionDecision"]


DESTRUCTIVE = [
    ("Bash", "rm -rf build/"),
    ("Bash", "cd x && rm -Rf ."),
    ("Bash", "git reset --hard origin/main"),
    ("Bash", "git push --force origin feature"),
    ("Bash", "git push -f"),
    ("Bash", "git clean -fdx"),
    ("Bash", "git checkout -- ."),
    ("Bash", "git branch -D old"),
    ("Bash", "git stash drop stash@{0}"),
    ("Bash", "git worktree remove --force .worktrees/x"),
    ("PowerShell", "Remove-Item -Recurse -Force C:\\temp\\x"),
    ("PowerShell", "rd /s /q build"),
    ("Bash", "sqlite3 db 'drop table users'"),
]

HARMLESS = [
    ("Bash", "git status --short"),
    ("Bash", "git branch -d merged-branch"),
    ("Bash", "git push -u origin feature"),
    ("Bash", "rm notes.txt"),
    ("Bash", "git checkout -b new-branch"),
    ("PowerShell", "Get-ChildItem -Recurse"),
    ("Bash", "echo format string"),
]


@pytest.mark.parametrize(("tool", "command"), DESTRUCTIVE)
def test_destructive_commands_ask(tool, command):
    assert _decision(tool, command) == "ask"


@pytest.mark.parametrize(("tool", "command"), HARMLESS)
def test_harmless_commands_pass_silently(tool, command):
    assert _decision(tool, command) is None


def test_other_tools_and_bad_payloads_are_never_blocked():
    assert _decision("Read", "rm -rf /") is None
    proc = _run("not json")
    assert proc.returncode == 0 and proc.stdout == ""
