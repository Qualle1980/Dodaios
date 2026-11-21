"""General (chatgpt-plan) supplies spoken content; ack_cfg.provider stays put."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.brain.factory import _flash_or_chat_seat, _flash_provider_label
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
