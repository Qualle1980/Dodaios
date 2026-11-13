"""Scripted ``subprocess.run`` stand-in for ``agy models`` auth probes.

Unit tests never shell out to a real ``agy`` binary (AGENTS.md: fakes over
mocks). Hand one of these to ``monkeypatch.setattr(auth_mod.subprocess, "run",
fake)`` so :func:`jarvis.google_cli.auth_service._probe_agy_subscription_via_models`
exercises its real parsing and cache logic against canned stdout.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeAgyModelsRun:
    """Return a canned ``CompletedProcess`` for ``agy models`` (or raise).

    Attributes:
        stdout: Text the fake CLI prints on success.
        returncode: Exit status (0 = success; empty stdout still means logged out).
        raise_exc: When set, ``run`` raises this instead of returning.
        calls: Argv lists observed, for cache / TTL assertions.
    """

    stdout: str = ""
    returncode: int = 0
    raise_exc: BaseException | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: list[str] | tuple[str, ...], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if self.raise_exc is not None:
            raise self.raise_exc
        return subprocess.CompletedProcess(
            list(argv),
            self.returncode,
            stdout=self.stdout,
            stderr="",
        )
