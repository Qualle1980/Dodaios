"""RunCommand must abort when timeout_s elapses (process tree included)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from jarvis.agent_chat.tools import execute_tool, normalize_command_timeout


def test_normalize_command_timeout_converts_millisecond_looking_values() -> None:
    assert normalize_command_timeout(120000) == 120.0
    assert normalize_command_timeout(30000) == 30.0
    assert normalize_command_timeout(None) == 120.0
    assert normalize_command_timeout(60) == 60.0
    assert normalize_command_timeout(900) == 900.0
    # Over the max and not a clean millisecond budget → cap, do not shrink to 1.8 s.
    assert normalize_command_timeout(1800) == 900.0
    assert normalize_command_timeout(5_000_000) == 900.0


@pytest.mark.asyncio
async def test_run_command_aborts_after_timeout_s(tmp_path: Path) -> None:
    """A sleep longer than timeout_s must return a timeout error quickly."""
    if sys.platform == "win32":
        # PowerShell/Git-Bash sleep: use Python so the shell choice does not matter.
        command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"
    started = time.perf_counter()
    out, err = await execute_tool(
        "RunCommand",
        {"command": command, "timeout_s": 2},
        cwd=tmp_path,
    )
    elapsed = time.perf_counter() - started
    assert err is True
    assert "timed out" in out.lower()
    assert "30" not in out or "timed out" in out.lower()
    # Must not wait for the full sleep; allow a little overhead for process kill.
    assert elapsed < 12.0, f"timeout took {elapsed:.1f}s; turn would block"


@pytest.mark.asyncio
async def test_run_command_millisecond_timeout_is_honoured(tmp_path: Path) -> None:
    """120000 must mean 120 s budget capped/normalised, not a 900 s hang."""
    # Normalisation alone is unit-tested; here ensure execute_tool uses it
    # by requesting a short ms-looking value (3000 → 3 s) against a longer sleep.
    if sys.platform == "win32":
        command = f'"{sys.executable}" -c "import time; time.sleep(20)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(20)'"
    started = time.perf_counter()
    out, err = await execute_tool(
        "RunCommand",
        {"command": command, "timeout_s": 3000},
        cwd=tmp_path,
    )
    elapsed = time.perf_counter() - started
    assert err is True
    assert "timed out after 3 s" in out.lower() or "timed out after 3 " in out.lower()
    assert elapsed < 12.0, f"ms-normalised timeout took {elapsed:.1f}s"
