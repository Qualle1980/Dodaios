"""The phase timer records durations and nothing else.

It exists to find the ~50 s of mission setup that no log accounts for, so the
two things that matter are that a slow phase is always reported — including
one that ends in an exception — and that the line carries no payload.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from jarvis.missions.timing import timed, timed_offload


def test_a_slow_phase_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.probe", "task-1"):
            time.sleep(0.08)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "setup.probe" in message
    assert "task-1" in message
    assert message.rstrip().endswith("ms")


def test_a_fast_phase_leaves_the_info_stream_unless_asked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.quick", "task-2"):
            pass
    assert caplog.records == []

    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.quick", "task-2", always=True):
            pass
    assert len(caplog.records) == 1


def test_a_fast_phase_is_still_recorded_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A phase is attributed by subtraction, so an absent line is not a zero."""
    with caplog.at_level(logging.DEBUG, logger="jarvis.missions.timing"):
        with timed("setup.quick", "task-6"):
            pass

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "setup.quick" in caplog.records[0].getMessage()


def test_a_failing_phase_is_timed_and_marked(caplog: pytest.LogCaptureFixture) -> None:
    """A step that fails after 40 s is exactly what this is hunting."""
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with pytest.raises(RuntimeError, match="boom"), timed("setup.broken", "task-3"):
            raise RuntimeError("boom")

    assert len(caplog.records) == 1
    assert "(failed)" in caplog.records[0].getMessage()


def test_the_exception_is_not_swallowed() -> None:
    with pytest.raises(ValueError, match="kept"), timed("setup.raises", "task-4"):
        raise ValueError("kept")


def test_nothing_but_phase_reference_and_duration_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No prompt, path, or identity may reach the log through this."""
    secret = "C:/Users/someone/private/prompt-text-and-a-token"
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.probe", "task-5", always=True):
            _ = secret

    line = caplog.records[0].getMessage()
    assert secret not in line
    assert line.startswith("mission timing: task-5 setup.probe")


def test_an_offload_reports_its_queue_wait_and_its_total(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The wait for a free thread is measured, never derived by subtraction."""
    def work(*, value: int) -> int:
        time.sleep(0.01)
        return value * 2

    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        result = asyncio.run(timed_offload("setup.offload", "task-7", work, value=21))

    assert result == 42
    phases = [r.getMessage().split()[3] for r in caplog.records]
    assert phases == ["setup.offload.queue_wait", "setup.offload"]
    assert all(r.levelno == logging.INFO for r in caplog.records)
    assert all("(failed)" not in r.getMessage() for r in caplog.records)


def test_an_offload_does_not_swallow_the_callables_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def work() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(timed_offload("setup.offload", "task-8", work))

    # The thread was entered, so the wait itself is a real number, not a bound.
    assert len(caplog.records) == 2
    assert all("(failed)" not in r.getMessage() for r in caplog.records)


def test_an_offload_cancelled_before_its_thread_marks_the_wait_as_a_bound() -> None:
    """A call that never reached a thread knows only a lower bound on its wait."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    async def scenario() -> None:
        started = asyncio.Event()

        def blocker() -> None:
            started.set()
            time.sleep(0.5)

        # One worker thread, already busy: the second call can only queue.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        busy = asyncio.create_task(asyncio.to_thread(blocker))
        await started.wait()

        queued = asyncio.create_task(timed_offload("setup.offload", "task-9", lambda: None))
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await busy

    handler = _Collect()
    logger = logging.getLogger("jarvis.missions.timing")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        asyncio.run(scenario())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    waits = [r for r in records if "setup.offload.queue_wait" in r.getMessage()]
    assert len(waits) == 1
    assert "(failed)" in waits[0].getMessage()
