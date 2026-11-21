"""General (chatgpt-plan) supplies spoken content; TTS stays Gemini."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.brain.factory import (
    _CHAT_SEAT,
    _flash_or_chat_seat,
    _flash_provider_label,
    build_spawn_announcer,
)
from jarvis.core.config import TTSConfig
from tests.unit.brain.test_ack_brain.conftest import make_ack_config


class _Seat:
    provider_name = "chatgpt-plan"


def test_flash_or_chat_seat_prefers_general_and_does_not_mutate_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seat = _Seat()
    monkeypatch.setattr(
        "jarvis.brain.factory._subscription_flash_provider",
        lambda _j, _a: (seat, "chatgpt-plan"),
    )
    built: list[str] = []

    def _flash(_j, _a):
        built.append("flash")
        return SimpleNamespace(name="gemini")

    monkeypatch.setattr("jarvis.brain.factory._build_flash_provider", _flash)

    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _flash_or_chat_seat(SimpleNamespace(), ack_cfg)

    assert provider is seat
    assert ack_cfg.provider == "follow_brain"
    assert built == []  # General wins; Flash REGISTRY not consulted
    assert _flash_provider_label(provider, ack_cfg) == "chatgpt-plan"


def test_flash_or_chat_seat_falls_back_to_registry_when_no_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jarvis.brain.factory._subscription_flash_provider",
        lambda _j, _a: (None, None),
    )
    flash = SimpleNamespace(name="gemini")
    monkeypatch.setattr(
        "jarvis.brain.factory._build_flash_provider",
        lambda _j, _a: flash,
    )
    ack_cfg = make_ack_config(provider="follow_brain")
    provider = _flash_or_chat_seat(SimpleNamespace(), ack_cfg)
    assert provider is flash
    assert ack_cfg.provider == "follow_brain"
    assert _flash_provider_label(provider, ack_cfg) == "gemini"


def test_spoken_content_seat_is_general_while_tts_default_is_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content composition uses chatgpt-plan; speech output defaults to Gemini TTS.

    Anchors:
    - content seat id: ``factory._CHAT_SEAT``
    - content wiring: ``factory._flash_or_chat_seat`` / ``build_spawn_announcer``
    - speech default: ``TTSConfig.provider == "gemini-flash-tts"``
    """
    assert _CHAT_SEAT == "chatgpt-plan"
    seat = _Seat()
    monkeypatch.setattr(
        "jarvis.brain.factory._subscription_flash_provider",
        lambda _j, _a: (seat, "chatgpt-plan"),
    )
    ack_cfg = make_ack_config(provider="gemini", enabled=True)
    if hasattr(ack_cfg, "model_copy"):
        ack_cfg = ack_cfg.model_copy(update={"spawn_announcements": True})
    jcfg = SimpleNamespace(ack_brain=ack_cfg)

    content = _flash_or_chat_seat(jcfg, ack_cfg)
    assert content is seat
    assert _flash_provider_label(content, ack_cfg) == "chatgpt-plan"

    announcer = build_spawn_announcer(jcfg)
    assert announcer._provider is seat

    assert TTSConfig().provider == "gemini-flash-tts"
    from jarvis.plugins.tts import build_tts_from_config

    selected: list[str] = []

    def _capture(cfg, name):
        selected.append(name)
        return SimpleNamespace(name=name, synthesize=None)

    monkeypatch.setattr("jarvis.plugins.tts._build_provider", _capture)
    monkeypatch.setattr(
        "jarvis.plugins.tts._resolve_keyed_tts_provider",
        lambda requested, cfg: (requested or "gemini-flash-tts", cfg),
    )
    built = build_tts_from_config(TTSConfig())
    assert selected and "gemini" in selected[0]
    assert built is not None
