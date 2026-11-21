"""A closing process tree must not abort the interpreter it is exiting with.

``ProcessTree.close()`` is synchronous and defers the ``SIGKILL`` escalation to
a daemon timer. A process that exits inside that grace window used to leave the
timer to fire while the interpreter was already finalizing; its ``logger.debug``
then wrote to a stderr sink mid-teardown and the process died with
``Fatal Python error: _enter_buffered_busy`` and exit code 250 — after every
test had already passed.

The crash itself is timing-dependent, so what is asserted here is the invariant
that removes it: every escalation stays reachable, and teardown both cancels it
and silences the sink. The end-to-end proof is the ``agent runtimes on
python:3.11-slim`` CI job, which reproduced the abort on this branch's head.
"""

from __future__ import annotations

import pytest

from jarvis.core import process_tree


@pytest.fixture(autouse=True)
def _clean_registry() -> object:
    """Leave the module's shutdown state exactly as it was found."""
    yield
    process_tree._stop_pending_escalations()
    with process_tree._pending_lock:
        process_tree._pending_timers.clear()
        process_tree._shutting_down = False


def _tree_with_group(monkeypatch: pytest.MonkeyPatch, signalled: list[int]) -> object:
    """A tree whose group signalling is recorded instead of sent to the OS."""
    monkeypatch.setattr(process_tree, "_killpg", lambda pgid, sig: signalled.append(sig))
    monkeypatch.setattr(process_tree, "_getpgid", lambda pid: pid)
    tree = process_tree._PosixProcessGroup("test-tree")
    tree.assign(4242)
    return tree


def test_close_registers_a_cancellable_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_tree, "GRACE_S", 30.0)
    signalled: list[int] = []
    tree = _tree_with_group(monkeypatch, signalled)

    tree.close()

    with process_tree._pending_lock:
        assert len(process_tree._pending_timers) == 1, "the escalation must be reachable"


def test_shutdown_cancels_the_escalation_before_it_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_tree, "GRACE_S", 30.0)
    signalled: list[int] = []
    tree = _tree_with_group(monkeypatch, signalled)
    tree.close()
    assert signalled == [process_tree.signal.SIGTERM]

    process_tree._stop_pending_escalations()

    with process_tree._pending_lock:
        assert not process_tree._pending_timers
    assert signalled == [process_tree.signal.SIGTERM], "no kill may run at teardown"


def test_close_during_shutdown_starts_no_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_tree, "GRACE_S", 30.0)
    signalled: list[int] = []
    tree = _tree_with_group(monkeypatch, signalled)
    process_tree._stop_pending_escalations()

    tree.close()

    with process_tree._pending_lock:
        assert not process_tree._pending_timers, "teardown must not gain a new thread"
    assert signalled == [process_tree.signal.SIGTERM], "the term still goes out"
