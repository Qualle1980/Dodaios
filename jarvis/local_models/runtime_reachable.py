"""Whether a keyless local brain runtime can actually answer.

``auth_mode == "none"`` only means "needs no API key". Selection used to treat
that as usable (AP-21 forbids gating on the absence of a credential). This
module asks the provider's own plugin for a free, read-only reachability
probe — typically a short local HTTP GET — and caches the answer so a burst of
worker/agents selections does not re-dial the same dead port.

A down runtime is an answer (``False``), never an exception on the boot or
selection path. Probing never starts a process, never downloads a model, and
never spends a token.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger(__name__)

__all__ = [
    "PROBE_TTL_S",
    "PROBE_TIMEOUT_S",
    "http_get_ok",
    "is_local_runtime_reachable",
    "reset_for_tests",
]

#: How long one reachability answer stands. Matches the Antigravity login
#: probe TTL in ``background_policy``: selection is asked in bursts, and a
#: 30 s window is short enough that starting Ollama is noticed on the next
#: background tick without re-probing every call while the owner waits.
PROBE_TTL_S = 30.0

#: Bound for one free HTTP reachability check. Tighter than the UI's
#: ``probe_host`` (3 s) and ``_server_version`` (1.5 s): this path runs while
#: a person or a background job is waiting for a provider pick.
PROBE_TIMEOUT_S = 0.5

_lock = threading.Lock()
_cache: dict[str, tuple[float, bool]] = {}
_inflight: set[str] = set()

#: Test seam: replaces the whole reachability answer for a provider.
_probe_override: Callable[[str], bool] | None = None

#: Test seam: ``httpx`` transport for :func:`http_get_ok` (FakeOllamaServer).
_http_transport: Any | None = None


def http_get_ok(
    url: str,
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    headers: dict[str, str] | None = None,
    transport: Any | None = None,
) -> bool:
    """True when ``url`` yields any HTTP response within ``timeout_s``.

    Connection refused, DNS failure and timeout are ``False``. A 4xx still
    counts as reachable — the runtime answered. Never raises. Read-only.
    """
    if not url.startswith(("http://", "https://")):
        return False
    chosen = transport if transport is not None else _http_transport
    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            transport=chosen,
        ) as client:
            client.get(url, headers=headers or {})
    except httpx.TimeoutException:
        # A black-holed port is "not usable", not a crash (AP-30: silence is
        # the answer the selection chain needs so it can walk on).
        return False
    except httpx.HTTPError:
        # Connection refused / DNS / protocol errors mean the runtime is down;
        # selection walks on (AP-30: False is the answer, not a crash).
        return False
    except Exception:  # noqa: BLE001 — probe glitch degrades to not usable
        log.debug("local-runtime probe of %s failed", url, exc_info=True)
        return False
    return True


def _plugin_reachable(provider: str) -> bool | None:
    """Ask the brain plugin for ``runtime_reachable``; ``None`` when undeclared."""
    from jarvis.core.entry_points import entry_points

    try:
        eps = [ep for ep in entry_points(group="jarvis.brain") if ep.name == provider]
        if not eps:
            return None
        cls = eps[0].load()
        probe = getattr(cls, "runtime_reachable", None)
        if not callable(probe):
            return None
        return bool(probe())
    except Exception:  # noqa: BLE001 — undeclared / broken probe → fall through
        log.debug(
            "local-runtime: plugin reachability probe for %s failed",
            provider,
            exc_info=True,
        )
        return None


def _generic_base_url_reachable(provider: str) -> bool | None:
    """Card-declared ``supports_base_url`` fallback when the plugin ships none.

    Resolves the configured / default root and GETs it. ``None`` when the card
    does not expose a server URL (nothing to probe without inventing a port).
    """
    from jarvis.core import config as cfg_mod
    from jarvis.ui.web.provider_spec import get_spec

    spec = get_spec(provider)
    if spec is None or not spec.supports_base_url:
        return None
    try:
        ep = cfg_mod.resolve_provider_endpoint(
            provider, vendor_default_base_url=spec.default_base_url
        )
    except Exception:  # noqa: BLE001 — unreadable endpoint is not reachable
        log.debug("local-runtime: endpoint for %s unreadable", provider, exc_info=True)
        return False
    root = str(ep.base_url or spec.default_base_url or "").rstrip("/")
    if not root:
        return False
    return http_get_ok(root)


def _probe_uncached(provider: str) -> bool:
    if _probe_override is not None:
        return bool(_probe_override(provider))
    declared = _plugin_reachable(provider)
    if declared is not None:
        return declared
    generic = _generic_base_url_reachable(provider)
    if generic is not None:
        return generic
    # Local billing but neither a plugin probe nor a base URL: nothing to
    # prove, so do not claim usable (AP-21 — capability, not keylessness).
    return False


def _store(provider: str, result: bool) -> bool:
    with _lock:
        _cache[provider] = (time.monotonic(), result)
        _inflight.discard(provider)
    return result


def _run_and_store(provider: str) -> bool:
    try:
        result = _probe_uncached(provider)
    except Exception:  # noqa: BLE001 — never raise into selection
        log.info(
            "local-runtime: reachability probe for %s crashed",
            provider,
            exc_info=True,
        )
        result = False
    return _store(provider, result)


def _schedule_background(provider: str) -> None:
    with _lock:
        if provider in _inflight:
            return
        _inflight.add(provider)

    thread = threading.Thread(
        target=_run_and_store,
        args=(provider,),
        name=f"local-runtime-probe-{provider}",
        daemon=True,
    )
    thread.start()


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # get_running_loop raises when no loop is running; that is the normal
        # off-loop answer, not a failure to report (AP-30).
        return False
    return True


def is_local_runtime_reachable(provider: str) -> bool:
    """Cached free reachability check for a keyless local brain provider.

    Off the event loop this waits (bounded) for a fresh answer. On the event
    loop it never blocks on I/O: a fresh cache hit is returned, otherwise the
    probe is scheduled on a daemon thread and this call answers ``False``
    (unproven is not usable) so the selection chain can walk on.
    """
    name = (provider or "").strip()
    if not name:
        return False
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit is not None and now - hit[0] < PROBE_TTL_S:
            return hit[1]

    if _on_event_loop():
        # Waiting here would freeze voice / WS (forensic 2026-07-27 class).
        # Unproven → not usable; the daemon fills the cache for the next pick.
        _schedule_background(name)
        return False
    return _run_and_store(name)


def reset_for_tests() -> None:
    """Drop cache, inflight markers and test seams."""
    global _probe_override, _http_transport
    with _lock:
        _cache.clear()
        _inflight.clear()
    _probe_override = None
    _http_transport = None
