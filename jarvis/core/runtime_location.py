"""Where this process is listening — facts for health and the voice tool.

Derived only from existing sources: ``JARVIS_BIND_HOST`` / the stamped listen
endpoint, ``jarvis.core.instance``, and ``surface_security.browser_login_required``.
No OS hostname, no secrets: ``/api/health`` stays a public, open_access surface.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from jarvis.core.instance import current_instance

#: Same loopback names ``control_auth.assert_bind_safe`` and the launcher use.
_LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_DEFAULT_ADMIN_PORT = 47821
BIND_HOST_ENV_VAR = "JARVIS_BIND_HOST"


def resolve_bind_host(environ: Mapping[str, str] | None = None) -> str:
    """The address this process intends to bind (env, else loopback default)."""
    source = os.environ if environ is None else environ
    raw = (source.get(BIND_HOST_ENV_VAR) or "127.0.0.1").strip()
    return raw or "127.0.0.1"


def is_loopback_bind(host: str) -> bool:
    """True when ``host`` is a loopback-only listen address."""
    normalised = (host or "").strip().lower().strip("[]")
    return normalised in _LOOPBACK_BIND_HOSTS


def _stamped_listen() -> tuple[str | None, int | None]:
    from jarvis.core import runtime_refs

    host = runtime_refs.get_listen_host()
    port = runtime_refs.get_listen_port()
    return host, port


def _port_from_api_base() -> int | None:
    from jarvis.core import runtime_refs

    base = runtime_refs.get_api_base_url()
    if not base:
        return None
    try:
        port = urlsplit(base).port
    except ValueError:
        # A malformed authority has no port to report; "unknown" is the honest
        # answer and the caller already treats None as "not listening yet".
        return None
    return int(port) if port else None


def resolve_listen_port(configured_port: int | None = None) -> int:
    """Effective admin port: stamped bind, then API base, then config / default."""
    _, stamped_port = _stamped_listen()
    if stamped_port is not None:
        return int(stamped_port)
    from_base = _port_from_api_base()
    if from_base is not None:
        return from_base
    if configured_port is not None:
        return int(configured_port)
    identity = current_instance()
    return identity.port(_DEFAULT_ADMIN_PORT) if not identity.is_default else _DEFAULT_ADMIN_PORT


def control_key_path_active(*, bind_host: str | None = None) -> bool:
    """True when the Control Key is the active boundary for some callers.

    Non-loopback binds always require it for remote clients. On loopback it is
    active only when the optional browser lock is ON
    (``surface_security.browser_login_required``).
    """
    from jarvis.ui.web.surface_security import browser_login_required

    host = resolve_bind_host() if bind_host is None else bind_host
    if not is_loopback_bind(host):
        return True
    return bool(browser_login_required())


def runtime_location_facts(
    *,
    version: str | None = None,
    configured_port: int | None = None,
    warming: bool = False,
) -> dict[str, Any]:
    """Public deployment facts for ``/api/health`` and the voice tool.

    Omits OS hostname and credentials. ``bind_host`` is the listen address
    (``127.0.0.1``, ``0.0.0.0``, …), which is operational, not a secret.
    """
    stamped_host, _ = _stamped_listen()
    bind_host = stamped_host or resolve_bind_host()
    loopback = is_loopback_bind(bind_host)
    identity = current_instance()
    payload: dict[str, Any] = {
        "ok": True,
        "instance": identity.name,
        "bind_host": bind_host,
        "bind_scope": "loopback" if loopback else "external",
        "port": resolve_listen_port(configured_port),
        "reachable_externally": not loopback,
        "control_key_path": control_key_path_active(bind_host=bind_host),
    }
    if version is not None:
        payload["version"] = version
    if warming:
        payload["warming"] = True
    return payload


__all__ = [
    "BIND_HOST_ENV_VAR",
    "control_key_path_active",
    "is_loopback_bind",
    "resolve_bind_host",
    "resolve_listen_port",
    "runtime_location_facts",
]
