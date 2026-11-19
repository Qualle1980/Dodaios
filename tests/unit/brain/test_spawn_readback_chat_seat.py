"""Spawn-Announcer and Readback-Composer use the chat seat (chatgpt-plan).

When Flash has no adapter for the plan primary, composition goes through
``resolve_subscription_brain(prefer="chatgpt-plan")`` — never Ollama, never a
keyed API family the user did not pick for spoken phrasing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.brain.ack_brain.providers.subscription import SubscriptionBrainFlashAck
from jarvis.brain.factory import (
    build_readback_composer,
    build_spawn_announcer,
)
from tests.unit.brain.test_ack_brain.conftest import make_ack_config


class _FakeSeatBrain:
    name = "chatgpt-plan"

    def complete(self, req: Any):  # noqa: ANN401
        async def _gen():
            yield SimpleNamespace(content="ok")
            return
            yield  # pragma: no cover — make this an async generator

        return _gen()


def _jcfg(*, primary: str = "chatgpt-plan") -> SimpleNamespace:
    return SimpleNamespace(
        ack_brain=make_ack_config(provider="follow_brain"),
        brain=SimpleNamespace(primary=primary),
    )


@pytest.fixture(autouse=True)
def _ollama_present_but_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "jarvis.brain.ollama_runtime.find_binary", lambda: "/usr/bin/ollama"
    )
    monkeypatch.setattr(
        "jarvis.core.config.get_secret", lambda key, env_fallback=None: "k-test"
    )


def test_spawn_announcer_chooses_chatgpt_plan_before_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    def _resolve(config, *, prefer=None, cli_timeout_s=None, bus=None):  # noqa: ANN001
        calls.append(prefer)
        return _FakeSeatBrain()

    monkeypatch.setattr(
        "jarvis.brain.resolver.resolve_subscription_brain", _resolve
    )

    composer = build_spawn_announcer(_jcfg(primary="chatgpt-plan"))

    assert calls == ["chatgpt-plan"]
    assert isinstance(composer._provider, SubscriptionBrainFlashAck)
    assert composer._provider.provider_name == "chatgpt-plan"
    assert type(composer._provider).__name__ != "OllamaFlashAck"
    assert getattr(composer._config, "provider", "") == "chatgpt-plan"


def test_readback_composer_chooses_chatgpt_plan_before_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    def _resolve(config, *, prefer=None, cli_timeout_s=None, bus=None):  # noqa: ANN001
        calls.append(prefer)
        return _FakeSeatBrain()

    monkeypatch.setattr(
        "jarvis.brain.resolver.resolve_subscription_brain", _resolve
    )

    composer = build_readback_composer(_jcfg(primary="chatgpt-plan"))

    assert calls == ["chatgpt-plan"]
    assert composer.has_llm is True
    assert isinstance(composer._provider, SubscriptionBrainFlashAck)
    assert composer._provider.provider_name == "chatgpt-plan"
    assert type(composer._provider).__name__ != "OllamaFlashAck"


def test_spawn_announcer_never_falls_to_ollama_without_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jarvis.brain.resolver.resolve_subscription_brain",
        lambda *a, **k: None,
    )

    composer = build_spawn_announcer(_jcfg(primary="chatgpt-plan"))

    assert composer._provider is None
    assert type(composer._provider).__name__ != "OllamaFlashAck"


def test_explicit_gemini_flash_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``[ack_brain].provider = gemini`` keeps the Flash adapter."""
    called = False

    def _resolve(*_a, **_k):  # noqa: ANN001
        nonlocal called
        called = True
        return _FakeSeatBrain()

    monkeypatch.setattr(
        "jarvis.brain.resolver.resolve_subscription_brain", _resolve
    )
    jcfg = _jcfg(primary="chatgpt-plan")
    jcfg.ack_brain = make_ack_config(provider="gemini")

    composer = build_spawn_announcer(jcfg)

    assert called is False
    assert type(composer._provider).__name__ == "GeminiFlashAck"
