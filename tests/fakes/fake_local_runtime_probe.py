"""Test seam for keyless-local reachability without dialling a real daemon.

Drives :mod:`jarvis.local_models.runtime_reachable` through its documented
override and optional ``httpx.MockTransport`` — never ``unittest.mock``, never
a real ``ollama`` process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from jarvis.local_models import runtime_reachable as reach

__all__ = [
    "install_http_transport",
    "install_override",
    "offline_transport",
    "online_transport",
    "reset",
    "timeout_transport",
]


def reset() -> None:
    reach.reset_for_tests()


def install_override(fn: Callable[[str], bool]) -> None:
    """Replace the whole reachability answer (per provider id)."""
    reach.reset_for_tests()
    reach._probe_override = fn


def install_http_transport(transport: httpx.MockTransport) -> None:
    """Route :func:`http_get_ok` through an in-process fake server."""
    reach.reset_for_tests()
    reach._http_transport = transport


def online_transport(
    *, path: str = "/api/version", body: dict[str, Any] | None = None
) -> httpx.MockTransport:
    payload = body if body is not None else {"version": "0.0-test"}

    def _handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == path:
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": f"no route for {request.url.path}"})

    return httpx.MockTransport(_handle)


def offline_transport() -> httpx.MockTransport:
    def _handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(_handle)


def timeout_transport() -> httpx.MockTransport:
    """Raise ``TimeoutException`` — the selection path treats that as down."""

    def _handle(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    return httpx.MockTransport(_handle)
