"""The CLI usage index is armed once after boot, without touching transcripts."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from jarvis.ui.web.costs_routes import _IndexRefresher, warm_index_after_boot


class _RecordingRefresher(_IndexRefresher):
    """A real refresher whose ``nudge`` only records the call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Path | None] = []
        self.fired = threading.Event()

    def nudge(self, data_dir: Path | None) -> None:  # type: ignore[override]
        self.calls.append(data_dir)
        self.fired.set()


def test_warm_arms_one_nudge_with_the_given_dir(tmp_path: Path) -> None:
    rec = _RecordingRefresher()
    timer = warm_index_after_boot(tmp_path, delay_s=0.0, refresher=rec)
    assert rec.fired.wait(timeout=5.0)
    timer.join(timeout=5.0)
    assert rec.calls == [tmp_path]


def test_warm_does_not_fire_before_the_delay(tmp_path: Path) -> None:
    rec = _RecordingRefresher()
    timer = warm_index_after_boot(tmp_path, delay_s=60.0, refresher=rec)
    time.sleep(0.05)
    assert rec.calls == []
    timer.cancel()
    timer.join(timeout=5.0)
    assert rec.calls == []


def test_warm_timer_is_a_daemon_so_it_never_holds_the_process(tmp_path: Path) -> None:
    rec = _RecordingRefresher()
    timer = warm_index_after_boot(tmp_path, delay_s=60.0, refresher=rec)
    try:
        assert timer.daemon is True
    finally:
        timer.cancel()
        timer.join(timeout=5.0)
