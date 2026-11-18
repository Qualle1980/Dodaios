"""The identity probe retries only the known transient launch error, once."""

from __future__ import annotations

import subprocess
from pathlib import Path

from jarvis.setup import macos_app_bundle as mab


def _completed(returncode: int, stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["open"], returncode=returncode, stdout="", stderr=stderr
    )


def _scripted(monkeypatch, results: list[subprocess.CompletedProcess[str]]) -> list[int]:
    calls: list[int] = []
    queue = list(results)

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        return queue.pop(0)

    monkeypatch.setattr(mab.subprocess, "run", fake_run)
    monkeypatch.setattr(mab.time, "sleep", lambda _s: None) if hasattr(mab, "time") else None
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(mab, "macos_launch_services_command", lambda *a, **k: ["open"])
    return calls


def test_transient_kevent_error_is_retried_once_then_succeeds(tmp_path: Path, monkeypatch) -> None:
    calls = _scripted(
        monkeypatch,
        [_completed(1, "Unable to block on applications (initial call to kevent() failed)"),
         _completed(0, "")],
    )
    result = mab._run_identity_probe(tmp_path / "x.app", tmp_path / "probe.json")
    assert result.returncode == 0
    assert len(calls) == 2


def test_other_failures_are_not_retried(tmp_path: Path, monkeypatch) -> None:
    calls = _scripted(monkeypatch, [_completed(1, "some other launch failure")])
    result = mab._run_identity_probe(tmp_path / "x.app", tmp_path / "probe.json")
    assert result.returncode == 1
    assert len(calls) == 1


def test_transient_error_twice_is_reported_not_hidden(tmp_path: Path, monkeypatch) -> None:
    calls = _scripted(
        monkeypatch,
        [_completed(1, "kevent() failed"), _completed(1, "kevent() failed")],
    )
    result = mab._run_identity_probe(tmp_path / "x.app", tmp_path / "probe.json")
    assert result.returncode == 1
    assert len(calls) == 2
