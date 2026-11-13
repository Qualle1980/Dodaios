"""Unit tests for deployment facts on /api/health and runtime_location."""

from __future__ import annotations

import pytest

from jarvis.core import runtime_refs
from jarvis.core.runtime_location import (
    control_key_path_active,
    is_loopback_bind,
    resolve_bind_host,
    runtime_location_facts,
)
from jarvis.ui.web import surface_security


@pytest.fixture(autouse=True)
def _clear_listen_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_refs, "_LISTEN_HOST", [])
    monkeypatch.setattr(runtime_refs, "_LISTEN_PORT", [])
    monkeypatch.setattr(runtime_refs, "_API_BASE_URL", [])
    surface_security.reset_browser_login_required()
    yield
    surface_security.reset_browser_login_required()


def test_loopback_bind_classification() -> None:
    assert is_loopback_bind("127.0.0.1")
    assert is_loopback_bind("localhost")
    assert is_loopback_bind("::1")
    assert not is_loopback_bind("0.0.0.0")
    assert not is_loopback_bind("192.168.1.10")


def test_resolve_bind_host_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_BIND_HOST", "0.0.0.0")
    assert resolve_bind_host() == "0.0.0.0"
    monkeypatch.delenv("JARVIS_BIND_HOST", raising=False)
    assert resolve_bind_host() == "127.0.0.1"


def test_control_key_path_on_loopback_follows_browser_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JARVIS_BIND_HOST", raising=False)
    surface_security.set_browser_login_required(False)
    assert control_key_path_active() is False
    surface_security.set_browser_login_required(True)
    assert control_key_path_active() is True


def test_external_bind_always_activates_control_key_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JARVIS_BIND_HOST", "0.0.0.0")
    surface_security.set_browser_login_required(False)
    assert control_key_path_active() is True


def test_facts_payload_shape_for_default_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVIS_BIND_HOST", raising=False)
    monkeypatch.delenv("JARVIS_INSTANCE", raising=False)
    surface_security.set_browser_login_required(False)
    facts = runtime_location_facts(version="9.9.9", configured_port=47821)
    assert facts["ok"] is True
    assert facts["version"] == "9.9.9"
    assert facts["instance"] == "default"
    assert facts["bind_host"] == "127.0.0.1"
    assert facts["bind_scope"] == "loopback"
    assert facts["port"] == 47821
    assert facts["reachable_externally"] is False
    assert facts["control_key_path"] is False
    assert "hostname" not in facts


def test_facts_prefer_stamped_listen_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_BIND_HOST", "127.0.0.1")
    runtime_refs.set_listen_endpoint("0.0.0.0", 48000)
    facts = runtime_location_facts()
    assert facts["bind_host"] == "0.0.0.0"
    assert facts["bind_scope"] == "external"
    assert facts["port"] == 48000
    assert facts["reachable_externally"] is True
    assert facts["control_key_path"] is True
