"""M6 fallback allowlist: switch only on connection/timeout, at most once.

Milestone (meilensteine-bis-release.md M6 / S6): a provider switch happens
only on connection errors (and timeouts), and at most once per turn. A model
error must not switch the provider.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import BrainRequest
from tests.fixtures.brain.fake_brain import FakeBrain


class _RaisingBrain:
    """Minimal brain that always raises ``error`` on ``complete``."""

    name = "raising-brain"
    context_window = 8192
    supports_tools = True
    supports_vision = False

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls: list[BrainRequest] = []

    async def complete(self, req: BrainRequest):  # type: ignore[no-untyped-def]
        self.calls.append(req)
        raise self.error
        yield  # pragma: no cover — keeps this an async generator

    def estimate_cost(self, req: BrainRequest) -> float:
        return 0.0


def _manager_with_chain(
    entries: list[tuple[str, str | None, Any]],
) -> tuple[BrainManager, dict[str, Any]]:
    """Wire ``BrainManager.generate`` to a fixed (provider, model, brain) chain."""
    bus = EventBus()
    manager = BrainManager(config=JarvisConfig(), bus=bus, tools={})
    manager._registry._loaded = True
    brains: dict[str, Any] = {}
    chain: list[tuple[str, str | None]] = []
    for name, model, brain in entries:
        manager._brain_cache[(name, model)] = brain
        brains[name] = brain
        chain.append((name, model))
    manager._build_fallback_chain = lambda level: list(chain)  # type: ignore[method-assign]
    return manager, brains


@pytest.mark.asyncio
async def test_model_error_does_not_switch_provider() -> None:
    """A 400 / rejected-prompt style failure must stay on the primary."""
    primary = _RaisingBrain(
        RuntimeError("Error code: 400 - invalid_request: prompt was rejected")
    )
    secondary = FakeBrain(text_response="stand-in must not answer")
    manager, brains = _manager_with_chain([
        ("gemini", "gemini-flash", primary),
        ("openrouter", "or-flash", secondary),
    ])

    result = await manager.generate("hi", use_history=False)

    assert primary.calls, "primary must be tried"
    assert brains["openrouter"].calls == [], (
        "model error must not switch to the next provider; "
        f"secondary was called {len(brains['openrouter'].calls)} time(s)"
    )
    assert "stand-in must not answer" not in result


@pytest.mark.asyncio
async def test_timeout_switches_exactly_once() -> None:
    """A timeout may cross to the next provider once; a second failure stops."""
    primary = _RaisingBrain(TimeoutError("Read timed out after 30s"))
    secondary = _RaisingBrain(TimeoutError("Read timed out after 30s"))
    tertiary = FakeBrain(text_response="third provider must not answer")
    manager, brains = _manager_with_chain([
        ("gemini", "gemini-flash", primary),
        ("openrouter", "or-flash", secondary),
        ("openai", "gpt-flash", tertiary),
    ])

    result = await manager.generate("hi", use_history=False)

    assert primary.calls, "primary must be tried"
    assert secondary.calls, "first timeout must switch exactly once"
    assert brains["openai"].calls == [], (
        "a second timeout must not switch again; "
        f"tertiary was called {len(brains['openai'].calls)} time(s)"
    )
    assert "third provider must not answer" not in result


@pytest.mark.asyncio
async def test_connection_error_switches_once_second_failure_does_not() -> None:
    """A connection error switches once; a second failure does not switch again."""
    primary = _RaisingBrain(
        RuntimeError("Connection error: [Errno 111] Connection refused")
    )
    secondary = _RaisingBrain(
        RuntimeError("Connection error: [Errno 111] Connection refused")
    )
    tertiary = FakeBrain(text_response="third provider must not answer")
    manager, brains = _manager_with_chain([
        ("gemini", "gemini-flash", primary),
        ("openrouter", "or-flash", secondary),
        ("openai", "gpt-flash", tertiary),
    ])

    result = await manager.generate("hi", use_history=False)

    assert primary.calls, "primary must be tried"
    assert secondary.calls, "first connection error must switch once"
    assert brains["openai"].calls == [], (
        "a second connection failure must not switch again; "
        f"tertiary was called {len(brains['openai'].calls)} time(s)"
    )
    assert "third provider must not answer" not in result
