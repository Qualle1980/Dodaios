"""Keyless local runtimes must prove they answer before selection (AP-21).

Defect (2026-11-10): ``is_default_usable("ollama")`` was True because
``auth_mode == "none"`` needs no key, while the daemon was refused on
127.0.0.1:11434 — background learning stayed dead after the worker-tier
rescue picked Ollama.
"""

from __future__ import annotations

import pytest

from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.core.model_selection import usable_worker_selection
from jarvis.local_models import assistant_session as policy
from jarvis.local_models import runtime_reachable as reach
from tests.fakes import fake_local_runtime_probe as fake_probe


@pytest.fixture(autouse=True)
def _clean_probe_state() -> None:
    fake_probe.reset()
    yield
    fake_probe.reset()


def _cfg(worker: str = "ollama", **fallback: str) -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.brain.primary = "gemini"
    cfg.brain.worker = BrainTierConfig(provider=worker, model=None, **fallback)
    return cfg


def test_reachable_local_runtime_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_probe.install_override(lambda p: p == "ollama")
    monkeypatch.setattr(policy, "is_tool_capable", lambda _p: True)
    monkeypatch.setattr(policy, "_known", lambda p: p in {"ollama", "gemini"})

    selected = usable_worker_selection(_cfg("ollama"))
    assert selected is not None
    assert selected.provider == "ollama"


def test_unreachable_local_runtime_is_skipped_for_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopped Ollama must not win; a keyed Gemini in the rescue tail does."""
    fake_probe.install_override(lambda _p: False)

    def usable(provider: str) -> bool:
        if provider == "ollama":
            return reach.is_local_runtime_reachable(provider)
        return provider == "gemini"

    monkeypatch.setattr(policy, "is_default_usable", usable)
    monkeypatch.setattr(policy, "is_tool_capable", lambda _p: True)
    monkeypatch.setattr(policy, "_known", lambda p: p in {"ollama", "gemini", "local-openai"})

    selected = usable_worker_selection(_cfg("ollama"))
    assert selected is not None
    assert selected.provider == "gemini"


def test_probe_is_not_repeated_within_ttl() -> None:
    calls: list[str] = []

    def _count(provider: str) -> bool:
        calls.append(provider)
        return True

    fake_probe.install_override(_count)
    assert reach.is_local_runtime_reachable("ollama") is True
    assert reach.is_local_runtime_reachable("ollama") is True
    assert calls == ["ollama"]


def test_probe_timeout_counts_as_not_usable() -> None:
    transport = fake_probe.timeout_transport()
    assert (
        reach.http_get_ok("http://ollama.test/api/version", transport=transport)
        is False
    )


def test_absent_runtime_never_raises() -> None:
    fake_probe.install_http_transport(fake_probe.offline_transport())
    assert reach.http_get_ok("http://127.0.0.1:11434/api/version") is False
    fake_probe.install_override(lambda _p: False)
    assert reach.is_local_runtime_reachable("ollama") is False
    assert policy.is_default_usable("ollama") is False


def test_is_default_usable_gates_on_local_reachability() -> None:
    fake_probe.install_override(lambda _p: False)
    assert policy.is_default_usable("ollama") is False

    fake_probe.reset()
    fake_probe.install_override(lambda _p: True)
    assert policy.is_default_usable("ollama") is True


def test_ttl_expires_and_reprobes() -> None:
    calls: list[str] = []
    state = {"ok": True}

    def _count(provider: str) -> bool:
        calls.append(provider)
        return bool(state["ok"])

    fake_probe.install_override(_count)
    assert reach.is_local_runtime_reachable("ollama") is True
    with reach._lock:
        stamped, value = reach._cache["ollama"]
        reach._cache["ollama"] = (stamped - reach.PROBE_TTL_S - 1.0, value)
    state["ok"] = False
    assert reach.is_local_runtime_reachable("ollama") is False
    assert calls == ["ollama", "ollama"]


def test_http_get_ok_online_via_fake_transport() -> None:
    fake_probe.install_http_transport(fake_probe.online_transport())
    assert reach.http_get_ok("http://ollama.test/api/version") is True


def test_ollama_plugin_runtime_reachable_uses_transport() -> None:
    from jarvis.plugins.brain.ollama import OllamaBrain

    fake_probe.install_http_transport(fake_probe.online_transport())
    assert OllamaBrain.runtime_reachable() is True

    fake_probe.reset()
    fake_probe.install_http_transport(fake_probe.offline_transport())
    assert OllamaBrain.runtime_reachable() is False
