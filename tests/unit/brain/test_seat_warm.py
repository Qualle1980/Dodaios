"""The first question should not pay for opening the conversation.

Measured 2026-10-20 against the running app: the FIRST question after a start
cost ~12 s of browser setup on top of the answer (effort menu 5.1 s, prompt
entry 2.1 s, send 5.0 s on a fresh conversation), while every later question
in the same session paid 1.5 s for the same three steps.

The warm-up is a provider call nobody asked for, so the guards matter more
than the feature: off unless the install switched it on, never for a provider
that bills a key, once per process, and advisory — a failure changes nothing
except that the first question pays what it pays today.

Deterministic — the brain is a recorder, nothing reaches a provider.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain import seat_warm
from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import BrainDelta


class _RecordingBrain:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def complete(self, req: Any):  # noqa: ANN201 - async generator
        self.requests.append(req)
        yield BrainDelta(content="ok")


class _Registry:
    def __init__(self) -> None:
        self.brain = _RecordingBrain()
        self.instantiated: list[str] = []

    def instantiate(self, name: str, **_k: Any) -> _RecordingBrain:
        self.instantiated.append(name)
        return self.brain


@pytest.fixture(autouse=True)
def _forget() -> None:
    seat_warm.reset_for_tests()


def _config(primary: str, *, on: bool) -> JarvisConfig:
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.routing.warm_seat_on_boot = on
    return config


async def test_a_plan_seat_is_opened_once() -> None:
    registry = _Registry()

    first = await seat_warm.warm_chat_seat(_config("chatgpt-web", on=True), registry, delay_s=0)
    second = await seat_warm.warm_chat_seat(_config("chatgpt-web", on=True), registry, delay_s=0)

    assert registry.instantiated == ["chatgpt-web"], "exactly one warm per process"
    assert "opened" in first and "already" in second
    assert registry.brain.requests[0].messages[0].content == seat_warm.WARM_PROMPT


async def test_it_does_nothing_unless_the_install_asked() -> None:
    registry = _Registry()

    outcome = await seat_warm.warm_chat_seat(_config("chatgpt-web", on=False), registry, delay_s=0)

    assert outcome == "off"
    assert registry.instantiated == []


@pytest.mark.parametrize("provider", ["openai", "claude-api", "gemini", "openrouter"])
async def test_a_key_billed_provider_is_never_warmed(provider: str) -> None:
    """The warm turn is one nobody asked for: it may never spend money."""
    registry = _Registry()

    outcome = await seat_warm.warm_chat_seat(_config(provider, on=True), registry, delay_s=0)

    assert registry.instantiated == []
    assert "not a plan seat" in outcome


async def test_the_switch_has_exactly_one_reader() -> None:
    assert seat_warm.warm_seat_enabled(_config("chatgpt-web", on=True)) is True
    assert seat_warm.warm_seat_enabled(_config("chatgpt-web", on=False)) is False


# --------------------------------------------------------------------------
# The wiring, not just the module.
# --------------------------------------------------------------------------

async def test_the_servers_hook_reaches_a_real_registry() -> None:
    """The first version imported a name the registry module does not have.

    The module tests passed — they hand in their own registry — and the task
    died with an ImportError that never reached the log, so the warm-up
    silently did nothing in the running app. This asserts the accessor the
    server actually uses exists and can be built.
    """
    from jarvis.brain.provider_registry import BrainProviderRegistry

    registry = BrainProviderRegistry()

    assert hasattr(registry, "instantiate")
    assert callable(registry.available)


async def test_the_attempt_waits_for_the_seat_to_come_up() -> None:
    """Scheduled at boot the attempt failed 0.2 s later: a browser-backed seat
    is not reachable that early. The wait is for a dependency, not a retry -
    after it the attempt is made exactly once."""
    registry = _Registry()
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    import asyncio as _asyncio

    real_sleep = _asyncio.sleep
    _asyncio.sleep = _record  # type: ignore[assignment]
    try:
        await seat_warm.warm_chat_seat(_config("chatgpt-web", on=True), registry)
    finally:
        _asyncio.sleep = real_sleep  # type: ignore[assignment]

    assert slept == [seat_warm.WARM_DELAY_S]
    assert registry.instantiated == ["chatgpt-web"]
