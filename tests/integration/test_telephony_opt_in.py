"""Telephony is mounted only when it is switched on.

The Twilio voice agent is an optional extra and ``integrations.twilio.enabled``
already defaults to false, yet its routes were mounted on every boot. That cost
the import of the telephony surface -- measured at 254 ms marginal with the
shared machinery already warm -- for a feature nobody had enabled.

Without the extra installed the router registers no routes at all (the
documented graceful degradation, AD-T8), so the manager is what tells the two
states apart everywhere; the route assertion runs only where the SDK exists.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig
from jarvis.ui.web.server import WebServer

_PREFIX = "/api/telephony"
_HAS_TWILIO = importlib.util.find_spec("twilio") is not None


def _server(*, telephony: bool) -> WebServer:
    cfg = JarvisConfig()
    cfg.ui.dev_mode = True
    cfg.integrations.twilio.enabled = telephony
    bus = EventBus()
    server = WebServer(cfg, bus=bus)
    server.app.state.config = cfg
    server.app.state.bus = bus
    return server


def _telephony_paths(server: WebServer) -> list[str]:
    return [
        path
        for route in server.app.router.routes
        if (path := getattr(route, "path", "")).startswith(_PREFIX)
    ]


def test_off_by_default_nothing_is_mounted() -> None:
    server = _server(telephony=False)
    assert _telephony_paths(server) == []
    # Honest state: switched off, not half-initialised.
    assert server.app.state.telephony_manager is None


def test_switching_it_on_initialises_it() -> None:
    assert _server(telephony=True).app.state.telephony_manager is not None


@pytest.mark.skipif(not _HAS_TWILIO, reason="the telephony extra is not installed")
def test_switching_it_on_mounts_the_routes() -> None:
    assert _telephony_paths(_server(telephony=True))


def test_the_direct_phone_path_does_not_depend_on_telephony() -> None:
    """PC-to-phone runs over the mobile gateway, which is its own application.

    Structural, not a route check: /api/m/call lives in
    ``jarvis.mobile_gateway.app``, a separate ASGI app, and importing it must
    not pull the telephony package in.
    """
    for name in [n for n in sys.modules if n.startswith("jarvis.telephony")]:
        del sys.modules[name]
    importlib.import_module("jarvis.mobile_gateway.app")
    assert not [n for n in sys.modules if n.startswith("jarvis.telephony")]
