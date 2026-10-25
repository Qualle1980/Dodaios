"""The phase timer records durations and nothing else.

It exists to find the ~50 s of mission setup that no log accounts for, so the
two things that matter are that a slow phase is always reported — including
one that ends in an exception — and that the line carries no payload.
"""
from __future__ import annotations

import logging
import time

import pytest

from jarvis.missions.timing import timed


def test_a_slow_phase_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.probe", "task-1"):
            time.sleep(0.08)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "setup.probe" in message
    assert "task-1" in message
    assert message.rstrip().endswith("ms")


def test_a_fast_phase_stays_quiet_unless_asked(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.quick", "task-2"):
            pass
    assert caplog.records == []

    with caplog.at_level(logging.INFO, logger="jarvis.missions.timing"):
        with timed("setup.quick", "task-2", always=True):
            pass
    assert len(caplog.records) == 1


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
