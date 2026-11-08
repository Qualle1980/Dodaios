"""One process writing ``count`` distinct keys into a config file via config_writer.

Used by ``test_config_write_race.py``. ``--nolock`` neutralises the cross-process
file lock so the test can prove it detects the regression it guards against;
without it the write path is exactly the product's.

Usage: config_write_race_worker.py <config path> <prefix> <count> [--nolock]
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

# Runs under ``-I``, which ignores PYTHONPATH, so an installed Dodaios would win
# over the checkout under test (AP-8). Point at this file's own tree instead.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> int:
    target, prefix, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    nolock = "--nolock" in sys.argv

    from jarvis.core import config_writer as cw

    if nolock:
        @contextmanager
        def no_file_lock(path: Path, timeout_s: float = 0.0):  # noqa: ARG001
            yield

        cw.file_lock.locked = no_file_lock  # type: ignore[assignment]

    failures = 0
    for i in range(count):
        try:
            cw._patch_table(target, "race", f"{prefix}_{i}", f"{prefix}-{i}")
        except Exception as exc:  # noqa: BLE001 - the test reports what the writer refused
            failures += 1
            print(f"{prefix}_{i}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
