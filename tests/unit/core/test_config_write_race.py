"""Two processes writing jarvis.toml at once must not lose each other's keys.

A ``threading.Lock`` cannot see another interpreter, and Dodaios runs several:
the desktop app, the ``jarvis`` CLI, agent sessions and mission workers all
write this file. Before ``_WriteGuard`` took an OS file lock, two of them doing
read-modify-write lost most of their keys AND reported success for every one of
them, because each wrote back a document it had read before the other's rename.

``test_nolock_loses_keys`` is the control: with the file lock neutralised the
loss comes back, so a green ``test_two_processes_lose_nothing`` means the lock
works and not that the test stopped looking.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("config_write_race_worker.py")
COUNT = 30
PREFIXES = ("a", "b")
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    target = tmp_path / "jarvis.toml"
    target.write_text('[brain]\nprimary = "test"\n\n[tts]\nvolume = 0.5\n', encoding="utf-8")
    return target


def _run_writers(target: Path, *extra: str) -> None:
    env = {**_base_env(), "PYTHONPATH": str(REPO_ROOT)}
    procs = [
        subprocess.Popen(  # noqa: S603 - fixed argv, test-local paths
            [sys.executable, "-I", str(WORKER), str(target), prefix, str(COUNT), *extra],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for prefix in PREFIXES
    ]
    for proc in procs:
        proc.communicate(timeout=300)


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _keys_on_disk(target: Path) -> set[str]:
    return set(tomllib.loads(target.read_text(encoding="utf-8-sig")).get("race", {}))


def _expected() -> set[str]:
    return {f"{prefix}_{i}" for prefix in PREFIXES for i in range(COUNT)}


def test_two_processes_lose_nothing(config_file: Path) -> None:
    _run_writers(config_file)

    missing = sorted(_expected() - _keys_on_disk(config_file))

    assert missing == [], (
        f"{len(missing)} keys written by one process were overwritten by the other"
    )


def test_the_other_sections_survive(config_file: Path) -> None:
    _run_writers(config_file)

    data = tomllib.loads(config_file.read_text(encoding="utf-8-sig"))

    assert data["brain"]["primary"] == "test"
    assert data["tts"]["volume"] == 0.5


def test_nolock_loses_keys(config_file: Path) -> None:
    """The control: without the file lock the losses return, so the test can see them."""
    _run_writers(config_file, "--nolock")

    missing = _expected() - _keys_on_disk(config_file)

    assert missing, (
        "no key was lost without the cross-process lock: this test can no longer "
        "detect the regression it guards, so the other two prove nothing"
    )
