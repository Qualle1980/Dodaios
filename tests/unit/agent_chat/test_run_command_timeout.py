"""RunCommand must abort when timeout_s elapses (process tree included)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import psutil
import pytest

from jarvis.agent_chat.tools import (
    _await_command_output,
    _run_command,
    execute_tool,
    normalize_command_timeout,
    shell_label,
)

#: Hard ceiling per acceptance case — a hang is the failure under test.
_CASE_LIMIT_S = 30.0
_STILL_ACTIVE = 259

#: Shell-syntax cases (``( … &)``, ``&``, ``>/dev/null``) need a POSIX shell:
#: Git Bash on Windows, bash/sh elsewhere. A PowerShell-only Windows box skips.
_needs_posix_shell = pytest.mark.skipif(
    shell_label() not in {"bash", "sh"},
    reason="needs a POSIX shell (Git Bash on Windows)",
)


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


def _unique_sleep_seconds() -> str:
    """A ``sleep`` duration no other process on the box is using.

    An orphan has no parent link and MSYS programs do not expose our
    environment to ``psutil``, so the command line is the one identity that
    survives; a random fraction makes it ours alone.
    """
    return f"{140 + uuid.uuid4().int % 50}.{uuid.uuid4().int % 997:03d}"


def _processes_with_arg(token: str) -> list[psutil.Process]:
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if proc.pid != os.getpid() and token in (proc.info["cmdline"] or []):
                found.append(proc)
        except psutil.Error:
            continue  # exited during the scan: not a survivor
    return found


async def _survivors_after_settle(token: str, settle_s: float = 2.0) -> list[psutil.Process]:
    """Processes still carrying ``token`` once a job close had time to land."""
    deadline = time.monotonic() + settle_s
    alive = _processes_with_arg(token)
    while alive and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        alive = _processes_with_arg(token)
    return alive


def _kill_all(procs: list[psutil.Process]) -> None:
    """Test hygiene: a failing case must not leave its sleepers on the box."""
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            continue  # already gone


class _DeafCommunicateProcess:
    """A process whose ``communicate()`` ignores cancellation until released.

    Stands in for a reader stuck on a pipe a descendant still holds open: the
    failure mode in which an unbounded ``await`` on the reader never returns.
    """

    def __init__(self) -> None:
        self.pid = 0  # no real process: the taskkill path must stay out of it
        self.returncode: int | None = None
        self.stdout = None
        self.stderr = None
        self.killed = False
        self.cancels_ignored = 0
        self.release = asyncio.Event()

    async def communicate(self) -> tuple[bytes, None]:
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancels_ignored += 1
        return b"", None

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = -9
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _RecordingTree:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


async def _finishes_within(coro, limit_s: float) -> tuple[bool, asyncio.Task]:
    """Run ``coro`` and report whether it finished, WITHOUT waiting on its cancel.

    ``asyncio.wait_for`` would cancel and then wait for a coroutine that ignores
    cancellation — exactly the hang under test — so the guard uses ``wait``.
    """
    task = asyncio.ensure_future(coro)
    done, _ = await asyncio.wait({task}, timeout=limit_s)
    return task in done, task


@pytest.mark.asyncio
async def test_timeout_returns_when_output_reader_ignores_cancellation() -> None:
    proc = _DeafCommunicateProcess()
    tree = _RecordingTree()
    started = time.perf_counter()
    finished, task = await _finishes_within(
        _await_command_output(proc, tree, 0.2), limit_s=12.0
    )
    elapsed = time.perf_counter() - started
    proc.release.set()  # let the deaf reader end so the loop tears down cleanly
    await asyncio.wait({task}, timeout=5.0)
    assert finished, "timeout path waited on a reader that ignores cancellation"
    assert task.result() == (None, True)
    assert proc.killed and tree.closed >= 1
    assert elapsed < 10.0, f"timeout path took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_cancel_returns_when_output_reader_ignores_cancellation() -> None:
    proc = _DeafCommunicateProcess()
    tree = _RecordingTree()
    task = asyncio.ensure_future(_await_command_output(proc, tree, 60.0))
    await asyncio.sleep(0.2)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=12.0)
    proc.release.set()
    await asyncio.wait({task}, timeout=5.0)
    assert task in done, "cancel path waited on a reader that ignores cancellation"
    assert task.cancelled(), "a cancelled turn must still see CancelledError"
    assert proc.killed and tree.closed >= 1


@_needs_posix_shell
@pytest.mark.asyncio
async def test_timeout_reaps_orphan_whose_subshell_already_exited(tmp_path: Path) -> None:
    """``(sleep N &)``: the subshell exits at once, its sleep has no parent left."""
    token = _unique_sleep_seconds()
    try:
        command = f"(sleep {token} &); sleep 150"
        out, err = await asyncio.wait_for(
            _run_command({"command": command, "timeout_s": 3}, cwd=tmp_path),
            timeout=_CASE_LIMIT_S,
        )
        assert err is True and "command timed out" in out.lower()
        survivors = await _survivors_after_settle(token)
        assert not survivors, f"orphaned sleep outlived the timeout: {[p.pid for p in survivors]}"
    finally:
        _kill_all(_processes_with_arg(token))


@_needs_posix_shell
@pytest.mark.asyncio
async def test_timeout_reaps_background_child(tmp_path: Path) -> None:
    token = _unique_sleep_seconds()
    try:
        command = f"sleep {token} & sleep {token}"
        out, err = await asyncio.wait_for(
            _run_command({"command": command, "timeout_s": 3}, cwd=tmp_path),
            timeout=_CASE_LIMIT_S,
        )
        assert err is True and "command timed out" in out.lower()
        survivors = await _survivors_after_settle(token)
        assert not survivors, f"background sleep outlived the timeout: {[p.pid for p in survivors]}"
    finally:
        _kill_all(_processes_with_arg(token))


@_needs_posix_shell
@pytest.mark.asyncio
async def test_finished_command_reaps_detached_background_child(tmp_path: Path) -> None:
    """The documented contract holds on success too: child processes end with the command."""
    token = _unique_sleep_seconds()
    try:
        started = time.perf_counter()
        out, err = await asyncio.wait_for(
            _run_command(
                # The pause lets the sleep really start: a job closed while the
                # subshell is still forking kills it before it exists, and the
                # test would pass without proving anything.
                {
                    "command": f"(sleep {token} >/dev/null 2>&1 &); sleep 1; echo started",
                    "timeout_s": 20,
                },
                cwd=tmp_path,
            ),
            timeout=_CASE_LIMIT_S,
        )
        assert err is False and "started" in out
        assert time.perf_counter() - started < 10.0
        survivors = await _survivors_after_settle(token)
        assert not survivors, f"detached sleep outlived its command: {[p.pid for p in survivors]}"
    finally:
        _kill_all(_processes_with_arg(token))


@_needs_posix_shell
@pytest.mark.asyncio
async def test_timeout_stops_find_over_a_small_tree(tmp_path: Path) -> None:
    """Live shape (a ``find`` walk) on a tree small enough to stay under 30 s."""
    walk_root = tmp_path / "walk"
    for i in range(30):
        (walk_root / f"d{i}").mkdir(parents=True)
        (walk_root / f"d{i}" / "f.txt").write_text("x", encoding="utf-8")
    token = _unique_sleep_seconds()
    command = f"find {walk_root.as_posix()} -type f -exec sleep {token} \\;"
    try:
        started = time.perf_counter()
        out, err = await asyncio.wait_for(
            _run_command({"command": command, "timeout_s": 3}, cwd=tmp_path),
            timeout=_CASE_LIMIT_S,
        )
        elapsed = time.perf_counter() - started
        assert err is True and "command timed out" in out.lower()
        assert elapsed <= 7.0, f"find timeout took {elapsed:.2f}s"
        survivors = await _survivors_after_settle(token)
        finds = [p for p in _processes_with_arg(walk_root.as_posix())]
        assert not survivors and not finds, (
            f"find tree outlived the timeout: sleep={[p.pid for p in survivors]} "
            f"find={[p.pid for p in finds]}"
        )
    finally:
        _kill_all(_processes_with_arg(token) + _processes_with_arg(walk_root.as_posix()))


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
        f"'import os,time; open(r\\\"{child_pid_path.as_posix()}\\\",\\\"w\\\")"
        ".write(str(os.getpid())); time.sleep(300)']);"
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
