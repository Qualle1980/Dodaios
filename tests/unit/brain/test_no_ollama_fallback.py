"""Ollama must never appear as an automatic fallback (Ahmad 2026-11-17).

The provider class may still exist for an explicit user pin. No resolve chain,
Flash ``follow_brain`` path, or seat successor may land on ``ollama`` on its
own — with the binary installed or not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import jarvis.core.config as cfg_mod
from jarvis.brain import resolver
from jarvis.brain.factory import (
    _build_flash_provider,
    _keyless_runtime_present,
    _pick_keyless_flash_fallback,
)
from jarvis.core.config import BrainConfig, JarvisConfig
from tests.unit.brain.test_ack_brain.conftest import make_ack_config


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    resolver._reset_for_tests()
    monkeypatch.setattr(cfg_mod, "get_provider_secret", lambda pid: None)
    import jarvis.brain.manager as manager

    monkeypatch.setattr(manager, "_keyless_provider_is_rescued_by_oauth", lambda pid: False)
    monkeypatch.setattr(
        "jarvis.brain.ollama_runtime.find_binary", lambda: "/usr/bin/ollama"
    )
    real_registry = resolver._get_registry()
    base_names = list(real_registry.available())
    for name in ("ollama", "local-openai"):
        if name not in base_names:
            base_names.append(name)

    class _RegistryWithOllama:
        """Pretend the Ollama brain entry-point is installed (it may not be here)."""

        def available(self) -> list[str]:
            return list(base_names)

    monkeypatch.setattr(resolver, "_get_registry", lambda: _RegistryWithOllama())
    yield
    resolver._reset_for_tests()


def test_resolve_chain_never_falls_back_to_ollama_when_ollama_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cfg_mod,
        "get_provider_secret",
        lambda pid: "sk-test" if pid in {"gemini", "openrouter"} else None,
    )
    config = JarvisConfig(brain=BrainConfig(primary="gemini"))

    providers = [p for p, _ in resolver._resolve_chain(config)]

    assert "ollama" not in providers
    assert "local-openai" in providers


def test_zero_key_chain_never_falls_back_to_ollama() -> None:
    providers = [p for p, _ in resolver._resolve_chain(JarvisConfig())]
    assert "ollama" not in providers
    assert "local-openai" in providers


def test_plan_seat_chain_never_falls_back_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = JarvisConfig(brain=BrainConfig(primary="chatgpt-plan"))
    providers = [p for p, _ in resolver._resolve_chain(config)]
    assert "ollama" not in providers


def test_flash_follow_brain_never_picks_ollama_for_plan_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "k-test"
    )
    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _build_flash_provider(
        SimpleNamespace(brain=SimpleNamespace(primary="chatgpt-plan")),
        ack_cfg,
    )
    assert provider is None
    assert ack_cfg.provider != "ollama"


def test_pick_keyless_flash_fallback_never_returns_ollama() -> None:
    ack_cfg = make_ack_config(provider="follow_brain")
    assert _pick_keyless_flash_fallback(ack_cfg) is None


def test_keyless_runtime_present_never_affirms_ollama() -> None:
    ack_cfg = make_ack_config(provider="follow_brain")
    assert _keyless_runtime_present(ack_cfg.providers.ollama) is False


def test_ollama_absent_from_ack_supported_fallback_list() -> None:
    from jarvis.brain.ack_brain.config import SUPPORTED_PROVIDERS

    assert "ollama" not in SUPPORTED_PROVIDERS
