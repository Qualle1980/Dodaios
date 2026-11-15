"""Parse ``agy models`` output — shared by the brain and the auth probe.

The Antigravity brain asks the CLI which models an account may run; the Google
CLI auth service uses the same parser as a capability gate when agy stores its
OAuth token in the OS keyring and writes no on-disk login file (AP-21).
"""
from __future__ import annotations


def parse_agy_models(stdout: str) -> tuple[str, ...]:
    """Model ids from ``agy models`` output (``id<TAB>display name`` per line).

    Tolerant on purpose: the command prints a progress line to stderr and may
    gain columns, so only the first tab-separated field of plausible lines is
    read.
    """
    ids: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        candidate = line.split("\t", 1)[0].strip()
        if candidate and " " not in candidate:
            ids.append(candidate)
    return tuple(ids)
