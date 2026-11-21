"""Scripted Google CLI auth service for route threading tests.

Unit tests never spawn the real ``agy`` / Gemini binary (AGENTS.md: fakes over
mocks). Hand one of these to ``monkeypatch.setattr(antigravity_routes,
"GoogleCliAuthService", lambda *a, **k: fake)`` so a route exercises its
``asyncio.to_thread`` offload against a controllable ``status()``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from jarvis.google_cli.auth_service import GoogleCliAuthStatus


@dataclass
class FakeGoogleCliAuthService:
    """Auth-service stand-in that records which thread ran ``status``.

    Attributes:
        status_value: Snapshot returned by ``status()``.
        status_block: Optional event the fake waits on inside ``status`` so a
            test can prove the asyncio loop stayed free while the probe ran.
        status_started: Set just before any optional block, so the test knows
            the probe entered the worker.
        status_threads: ``threading.get_ident()`` values observed in ``status``.
        start_login_pid: Pid reported by ``start_login``.
        start_login_raises: When set, ``start_login`` raises this instead.
        logout_ok: Pair returned by ``logout_blocking``.
    """

    status_value: GoogleCliAuthStatus = field(
        default_factory=lambda: GoogleCliAuthStatus(
            installed=True, connected=False, mode="none", message="not connected"
        )
    )
    status_block: threading.Event | None = None
    status_started: threading.Event = field(default_factory=threading.Event)
    status_threads: list[int] = field(default_factory=list)
    start_login_pid: int = 999
    start_login_raises: BaseException | None = None
    logout_ok: tuple[bool, str | None] = (True, None)

    def status(self) -> GoogleCliAuthStatus:
        self.status_threads.append(threading.get_ident())
        self.status_started.set()
        if self.status_block is not None:
            # A blocked probe on the event loop freezes every other await.
            # No timeout: the test must release this, or hang until pytest kills it.
            self.status_block.wait()
        return self.status_value

    def start_login(self) -> Any:
        if self.start_login_raises is not None:
            raise self.start_login_raises

        class _Proc:
            pid = self.start_login_pid

        return _Proc()

    def logout_blocking(self) -> tuple[bool, str | None]:
        return self.logout_ok
