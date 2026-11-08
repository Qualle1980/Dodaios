"""One exclusive lock on one file, held across processes.

A ``threading.Lock`` serialises the writers inside one interpreter and nothing
else. Dodaios runs more than one: the desktop app, the ``jarvis`` CLI, agent
sessions and mission workers all write ``jarvis.toml``. Two of them doing
read-modify-write at the same time lose each other's keys and both report
success, because each one writes a whole document it read before the other's
rename.

So the lock has to be the operating system's: ``msvcrt.locking`` of the first
byte on Windows, ``fcntl.flock`` elsewhere. The OS drops it when the holder
dies, so a crash cannot leave the next writer waiting forever.

The lock lives beside the file it guards as ``<name>.lock`` and is never the
file itself: locking the target would collide with the atomic rename that
replaces it.

This is the primitive :class:`jarvis.agent_runtimes.base.HomeFileLock` already
used for runtime folders, lifted into ``core`` so the config writers can reach
it without importing a higher layer.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

#: How long a writer waits for another process before giving up. Long enough to
#: outlast a normal write (milliseconds), short enough that a wedged holder
#: surfaces as an error instead of a hang.
DEFAULT_TIMEOUT_S = 10.0

_POLL_S = 0.01


def lock_path_for(target: Path) -> Path:
    """The lock file guarding ``target``."""
    return target.with_name(target.name + ".lock")


class FileLock:
    """An exclusive, cross-process lock on ``path``.

    Not reentrant: a second :meth:`acquire` from the same thread while the lock
    is held would deadlock against itself on POSIX, so callers hold it once
    around the whole read-modify-write.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Take the lock if it is free. ``False`` means someone else holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # msvcrt locks a byte RANGE, so the offset must be deterministic.
            os.lseek(fd, 0, os.SEEK_SET)
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415 - Windows only

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415 - POSIX only

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:  # held elsewhere: the caller retries or gives up
            os.close(fd)
            return False
        self._fd = fd
        return True

    def acquire(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Block until the lock is free. Raises :class:`TimeoutError` after ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        while not self.try_acquire():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"another process holds {self.path} after {timeout_s}s")
            time.sleep(_POLL_S)

    async def acquire_async(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """:meth:`acquire` without blocking the event loop."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not self.try_acquire():
            if loop.time() >= deadline:
                raise TimeoutError(f"another process holds {self.path} after {timeout_s}s")
            await asyncio.sleep(_POLL_S)

    def release(self) -> None:
        """Let the lock go. Safe to call when it is not held."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415 - Windows only

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415 - POSIX only

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:  # closing the descriptor below releases it anyway
            log.debug("file lock: unlocking %s failed: %s", self.path, exc)
        finally:
            os.close(fd)


@contextmanager
def locked(target: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Iterator[None]:
    """Hold the cross-process lock for ``target`` for the duration of the block.

    ``target`` is the guarded file; the lock itself is ``<name>.lock`` beside it.
    """
    lock = FileLock(lock_path_for(target))
    lock.acquire(timeout_s)
    try:
        yield
    finally:
        lock.release()
