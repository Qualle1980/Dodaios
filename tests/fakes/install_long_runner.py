"""A harmless long-running process whose executable lives inside a fake install.

Two uninstall tests need the exact condition that broke the real uninstall: a
process still executing out of the install directory, holding its files. Both
built that stand-in by copying a system long-runner (``ping`` on Windows,
``sleep`` on POSIX) into the throwaway tree and starting it from there.

The POSIX half of that stopped working when ``ubuntu-latest`` moved to Ubuntu
26.04, which ships **rust-coreutils (uutils)** as the default provider. Its
utilities are one MULTICALL binary that decides what to do from the name it is
called by, so a copy named ``jarvis-fake`` recognises nothing and exits 1
immediately. The tests then asserted their own premise away: nothing was
running when the stop step ran.

A shell does not care what it is called, so it is the stand-in that holds
across both coreutils implementations. The command keeps two statements on
purpose: ``sh -c 'sleep N'`` is a single simple command, which dash execs
directly, and the in-tree executable would be replaced by ``/usr/bin/sleep``
from outside the tree -- the opposite of what these tests set up.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Long enough to outlive the stop step, short enough to never linger in CI.
_SECONDS = 120


def bin_dir_for(install_dir: Path) -> Path:
    """The venv script directory the uninstall bootstraps look at."""
    return install_dir / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")


def start_long_runner(install_dir: Path, seconds: int = _SECONDS) -> subprocess.Popen[bytes]:
    """Start a process whose executable sits INSIDE ``install_dir``.

    Raises ``OSError`` when the stand-in cannot be copied or started, which the
    callers turn into a skip: a hardened host without copy rights is not a
    product defect.
    """
    bin_dir = bin_dir_for(install_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # ping.exe is not a multicall binary; a copy under any name still pings.
        source = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "ping.exe"
        target = bin_dir / "jarvis-fake.exe"
        shutil.copy2(source, target)
        return subprocess.Popen(  # noqa: S603 - self-copied system binary
            [str(target), "-n", str(seconds), "127.0.0.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    shell = shutil.which("sh")
    if shell is None:
        raise OSError("no POSIX shell to copy as the stand-in long-runner")
    target = bin_dir / "jarvis-fake"
    # copy, not copy2: the macOS system shell carries SIP flags a copy must not.
    shutil.copy(shell, target)
    target.chmod(0o755)
    return subprocess.Popen(  # noqa: S603 - self-copied system shell
        [str(target), "-c", f"sleep {seconds}; exit 0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
