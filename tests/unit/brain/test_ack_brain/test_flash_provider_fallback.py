"""Tests for the Flash-Brain's ``follow_brain`` fallback.

When ``brain.primary`` points at a provider the Flash-Brain has no adapter
for (openrouter, claude_api, a plan seat), ``_build_flash_provider`` used to
pick the first REGISTRY family with a stored API key. That billed a key the
user never chose for spoken phrasing (live 2026-10-23: a ``chatgpt-plan``
install composed every spawn announcement on its Gemini key). The fallback is
now keyless only; with no local runtime the flash tier returns ``None`` and
its consumers speak their canned lines.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.brain.factory import _build_flash_provider
from tests.unit.brain.test_ack_brain.conftest import make_ack_config


def _jcfg(*, brain_primary: str) -> SimpleNamespace:
    return SimpleNamespace(brain=SimpleNamespace(primary=brain_primary))


@pytest.fixture(autouse=True)
def _ollama_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run as if the local runtime is installed unless they say not."""
    monkeypatch.setattr(
        "jarvis.brain.ollama_runtime.find_binary", lambda: "/usr/bin/ollama"
    )
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_an_uninstalled_local_runtime_is_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Ollama binary and a localhost endpoint: canned lines, no runtime."""
    monkeypatch.setattr("jarvis.brain.ollama_runtime.find_binary", lambda: "")
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "k-test"
    )

    ack_cfg = make_ack_config(provider="follow_brain")
    assert _build_flash_provider(_jcfg(brain_primary="chatgpt-plan"), ack_cfg) is None


def test_a_remote_runtime_counts_without_a_local_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jarvis.brain.ollama_runtime.find_binary", lambda: "")
    ack_cfg = make_ack_config(provider="follow_brain")
    ack_cfg.providers.ollama.endpoint = "http://192.168.1.20:11434"

    _build_flash_provider(_jcfg(brain_primary="chatgpt-plan"), ack_cfg)

    assert ack_cfg.provider == "ollama"


@pytest.mark.parametrize(
    "key_name", ["openai_api_key", "gemini_api_key", "realtime_gemini_api_key"]
)
def test_a_stored_key_never_becomes_the_flash_fallback(
    monkeypatch: pytest.MonkeyPatch, key_name: str
) -> None:
    """A primary without a Flash adapter resolves to the keyless local
    runtime even though another family has a usable key."""
    monkeypatch.setattr(
        "jarvis.core.config.get_secret",
        lambda key, env_fallback=None: "k-test" if key == key_name else None,
    )

    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _build_flash_provider(_jcfg(brain_primary="chatgpt-plan"), ack_cfg)

    assert ack_cfg.provider == "ollama"
    assert type(provider).__name__ == "OllamaFlashAck"


def test_no_keyless_runtime_means_canned_lines_not_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a keyless REGISTRY family the flash tier is off (``None``),
    even with every key stored."""
    from jarvis.brain.ack_brain import providers

    keyed = {k: v for k, v in providers.REGISTRY.items() if k != "ollama"}
    monkeypatch.setattr(providers, "REGISTRY", keyed)
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "k-test"
    )

    ack_cfg = make_ack_config(provider="follow_brain")
    assert _build_flash_provider(_jcfg(brain_primary="claude-api"), ack_cfg) is None
    assert ack_cfg.provider == "follow_brain"


def test_an_explicit_flash_provider_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider the user set in ``[ack_brain].provider`` is their choice."""
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "gk-test"
    )

    ack_cfg = make_ack_config(provider="gemini")
    provider = _build_flash_provider(_jcfg(brain_primary="chatgpt-plan"), ack_cfg)

    assert type(provider).__name__ == "GeminiFlashAck"


def test_does_not_change_ack_behavior_when_primary_has_a_flash_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When brain.primary IS a REGISTRY family, the fallback path is never
    consulted — no credential lookup happens at all."""
    called = False

    def _fail_get_secret(key: str, env_fallback: str | None = None) -> str | None:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr("jarvis.core.config.get_secret", _fail_get_secret)

    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _build_flash_provider(_jcfg(brain_primary="openai"), ack_cfg)

    assert ack_cfg.provider == "openai"
    assert type(provider).__name__ == "OpenAIMiniAck"
    assert called is False


def _live_jcfg(*, brain_primary: str) -> SimpleNamespace:
    """GPT-Live is the voice: the OpenAI key belongs to the voice call."""
    return SimpleNamespace(
        voice=SimpleNamespace(mode="realtime"),
        brain=SimpleNamespace(
            primary=brain_primary,
            realtime=SimpleNamespace(provider="openai-live"),
        ),
    )


def test_primary_on_the_voice_key_never_moves_flash_work_to_another_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mandate 2026-09-29: the GPT-Live key pays for the voice call only, and
    another family's key is not a substitute."""
    keys = {"openai_api_key": "sk-test", "gemini_api_key": "gk-test"}
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: keys.get(key)
    )

    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _build_flash_provider(_live_jcfg(brain_primary="openai"), ack_cfg)

    assert ack_cfg.provider == "ollama"
    assert type(provider).__name__ == "OllamaFlashAck"


def test_single_voice_key_install_does_not_bill_the_voice_key_for_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The voice key is not reused for flash work; the keyless runtime is
    tried and a missing one degrades to canned lines."""
    monkeypatch.setattr(
        "jarvis.core.config.get_secret",
        lambda key, env_fallback=None: "sk-test" if key == "openai_api_key" else None,
    )

    ack_cfg = make_ack_config(provider="follow_brain")
    _build_flash_provider(_live_jcfg(brain_primary="openai"), ack_cfg)

    assert ack_cfg.provider == "ollama"


def test_pipeline_voice_keeps_following_an_openai_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a realtime voice the key is an ordinary brain key."""
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "sk-test"
    )
    jcfg = _live_jcfg(brain_primary="openai")
    jcfg.voice.mode = "pipeline"

    ack_cfg = make_ack_config(provider="follow_brain")
    _build_flash_provider(jcfg, ack_cfg)

    assert ack_cfg.provider == "openai"
