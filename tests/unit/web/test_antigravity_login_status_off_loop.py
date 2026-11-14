"""``POST /antigravity/login`` must offload ``status()`` off the event loop.

After the agy keyring probe, ``GoogleCliAuthService.status()`` can block for
up to several seconds when no on-disk token is present — which is exactly the
state a user is in when they press Connect with Google. Running that probe on
the asyncio loop freezes the realtime voice socket (forensic 2026-07-27).
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from jarvis.google_cli.auth_service import GoogleCliAuthStatus
from jarvis.ui.web import antigravity_routes
from tests.fakes.fake_google_cli_auth_service import FakeGoogleCliAuthService


@pytest.mark.asyncio
async def test_antigravity_login_runs_status_off_the_event_loop(monkeypatch) -> None:
    """A blocked status probe must leave the event loop free to schedule work."""
    loop_thread = threading.get_ident()
    release = threading.Event()
    fake = FakeGoogleCliAuthService(
        status_value=GoogleCliAuthStatus(
            installed=True, connected=False, mode="none", message="not connected"
        ),
        status_block=release,
    )
    monkeypatch.setattr(
        antigravity_routes, "GoogleCliAuthService", lambda *a, **k: fake
    )

    task = asyncio.create_task(antigravity_routes.antigravity_login())
    try:
        # Wait until the probe has entered status() without holding the loop.
        deadline = time.monotonic() + 2.0
        while not fake.status_started.is_set():
            if time.monotonic() > deadline:
                pytest.fail("status() never started")
            await asyncio.sleep(0.01)

        # If status() were running on this loop, release.wait() would freeze it
        # and this sleep would never complete.
        t0 = time.monotonic()
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=0.5)
        assert time.monotonic() - t0 < 0.4, (
            "event loop stalled while status() was in flight — probe ran inline"
        )

        assert fake.status_threads, "status() was never called"
        assert fake.status_threads[0] != loop_thread, (
            f"status() ran on the event-loop thread ({loop_thread}); "
            "it must be awaited via asyncio.to_thread"
        )
    finally:
        release.set()

    result = await asyncio.wait_for(task, timeout=2)
    assert result["ok"] is True
    assert result["pid"] == 999
