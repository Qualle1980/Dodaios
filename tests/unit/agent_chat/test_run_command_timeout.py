"""RunCommand must abort when timeout_s elapses (process tree included)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from jarvis.agent_chat.tools import _run_command, execute_tool, normalize_command_timeout

#: Hard ceiling per acceptance case — a hang is the failure under test.
_CASE_LIMIT_S = 30.0
_STILL_ACTIVE = 259


def _pid_alive(pid: int) -> bool:
    """True when the OS still has a live process for ``pid``."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == _STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


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


@pytest.mark.asyncio
async def test_run_command_timeout_kills_grandchild_holding_stdout(tmp_path: Path) -> None:
    """Shell→Python→child; child inherits stdout and sleeps. Must die with the timeout."""
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    # Child keeps the inherited stdout pipe open while sleeping.
    command = (
        f'"{sys.executable}" -c '
        "\"import os,subprocess,sys,time;"
        f"open(r'{parent_pid_path.as_posix()}','w').write(str(os.getpid()));"
        "p=subprocess.Popen([sys.executable,'-c',"
        f"'import os,time; open(r\\\"{child_pid_path.as_posix()}\\\",\\\"w\\\").write(str(os.getpid())); time.sleep(300)']);"
        "time.sleep(300)\""
    )
    started = time.perf_counter()
    out, err = await asyncio.wait_for(
        _run_command({"command": command, "timeout_s": 3}, cwd=tmp_path),
        timeout=_CASE_LIMIT_S,
    )
    elapsed = time.perf_counter() - started
    assert err is True
    assert "command timed out" in out.lower()
    assert 3.0 <= elapsed <= 6.0, f"expected return in 3–6 s, got {elapsed:.2f}s"
    # Pid files must exist (command actually started) and both processes must be dead.
    assert parent_pid_path.is_file(), "parent never wrote its pid"
    assert child_pid_path.is_file(), "grandchild never wrote its pid"
    parent_pid = _read_pid(parent_pid_path)
    child_pid = _read_pid(child_pid_path)
    # Brief settle for Windows handle teardown after taskkill.
    await asyncio.sleep(0.4)
    assert not _pid_alive(parent_pid), f"parent pid {parent_pid} still alive after timeout"
    assert not _pid_alive(child_pid), f"grandchild pid {child_pid} still alive after timeout"


@pytest.mark.asyncio
async def test_run_command_timeout_heredoc_walk(tmp_path: Path) -> None:
    """Livetest shape: ``python - <<'PY'`` with os.walk; must abort and reap children."""
    # Build a tree large enough that a walk with a tiny sleep outlives timeout_s=3.
    walk_root = tmp_path / "tree"
    for i in range(40):
        d = walk_root / f"d{i}"
        d.mkdir(parents=True)
        for j in range(20):
            (d / f"f{j}.txt").write_text("x", encoding="utf-8")
    pid_path = tmp_path / "walk.pid"
    command = f"""python - <<'PY'
import os, time
root = r'{walk_root.as_posix()}'
open(r'{pid_path.as_posix()}', 'w').write(str(os.getpid()))
n = 0
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        n += 1
        time.sleep(0.05)
        if n % 10 == 0:
            print(n, flush=True)
print('done', n)
PY"""
    started = time.perf_counter()
    out, err = await asyncio.wait_for(
        _run_command({"command": command, "timeout_s": 3}, cwd=tmp_path),
        timeout=_CASE_LIMIT_S,
    )
    elapsed = time.perf_counter() - started
    assert err is True
    assert "command timed out" in out.lower()
    assert elapsed <= 6.0, f"heredoc walk timeout took {elapsed:.2f}s"
    assert pid_path.is_file(), "heredoc python never wrote its pid"
    walk_pid = _read_pid(pid_path)
    await asyncio.sleep(0.4)
    assert not _pid_alive(walk_pid), f"heredoc python pid {walk_pid} still alive after timeout"


@pytest.mark.asyncio
async def test_run_command_fast_success(tmp_path: Path) -> None:
    """A quick command still returns stdout and a non-error exit footer."""
    command = f'"{sys.executable}" -c "print(42)"'
    started = time.perf_counter()
    out, err = await asyncio.wait_for(
        _run_command({"command": command, "timeout_s": 10}, cwd=tmp_path),
        timeout=_CASE_LIMIT_S,
    )
    elapsed = time.perf_counter() - started
    assert err is False
    assert "42" in out
    assert "[exit 0" in out
    assert elapsed < 10.0


@pytest.mark.asyncio
async def test_run_command_finishes_exactly_at_timeout_boundary(tmp_path: Path) -> None:
    """A command that ends near timeout_s must return once — no double result, no hang."""
    # Sleep just under the budget so success wins the race cleanly most of the time;
    # the important property is a single return within the case limit.
    command = f'"{sys.executable}" -c "import time; time.sleep(1.5); print(\'edge-done\')"'
    started = time.perf_counter()
    out, err = await asyncio.wait_for(
        _run_command({"command": command, "timeout_s": 2}, cwd=tmp_path),
        timeout=_CASE_LIMIT_S,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 8.0, f"boundary case hung or double-waited ({elapsed:.2f}s)"
    # Either finished successfully or timed out once — never both payloads.
    lower = out.lower()
    timed_out = "command timed out" in lower
    finished = "edge-done" in out and "[exit 0" in out
    assert timed_out ^ finished, f"expected exactly one outcome, got: {out!r}"
    if timed_out:
        assert err is True
    else:
        assert err is False
