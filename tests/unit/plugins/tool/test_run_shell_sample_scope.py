"""run_shell tells the model when a command capped its own result set.

Live 2026-11-15 a seat ran ``gh pr list --limit 3`` through run_shell, saw
three merged PRs and answered that there were no open PRs; thirteen were open.
"""
from __future__ import annotations

import sys

from jarvis.plugins.tool.run_shell import RunShellTool


async def _run(command: str):
    return await RunShellTool().execute({"command": command}, ctx=None)


async def test_a_limited_command_is_marked_as_a_sample():
    echo = "Write-Output 'pr 1'" if sys.platform == "win32" else "echo 'pr 1'"
    result = await _run(f"{echo} # gh pr list --limit 3")
    assert result.output["scope"] == "(sample of 3; not the complete set)"


async def test_an_uncapped_command_carries_no_scope():
    echo = "Write-Output 'pr 1'" if sys.platform == "win32" else "echo 'pr 1'"
    result = await _run(echo)
    assert "scope" not in result.output
