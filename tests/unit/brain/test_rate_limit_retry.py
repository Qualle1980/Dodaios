"""429 retryDelay: parse, wait once, retry, then fallback."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.brain.rate_limit_tracker import (
    RateLimitTracker,
    clamp_retry_wait,
    parse_retry_delay,
    plan_rate_limit_retry,
)


def test_parse_retry_delay_from_google_error():
    msg = (
        "429 RESOURCE_EXHAUSTED. "
        "{'error': {'code': 429, 'message': '...', "
        "'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
        "\"retryDelay\": \"28s\"}]}}"
    )
    assert parse_retry_delay(msg) == 28.0


def test_parse_retry_delay_default_when_absent():
    assert parse_retry_delay("something else", default=30.0) == 30.0


def test_clamp_retry_wait_caps_hour_long_quota():
    assert clamp_retry_wait(3600) == 300.0
    assert clamp_retry_wait(-1) == 0.0


def test_plan_waits_once_then_falls_back():
    retried: set[tuple[str, str | None]] = set()
    key = ("gemini", "flash")
    msg = "retryDelay: '12s'"
    wait, delay = plan_rate_limit_retry(key, retried, msg)
    assert wait is True and delay == 12.0 and key in retried
    wait2, delay2 = plan_rate_limit_retry(key, retried, msg)
    assert wait2 is False and delay2 == 12.0


@pytest.mark.asyncio
async def test_manager_rate_limit_waits_once_then_marks(monkeypatch: pytest.MonkeyPatch):
    """Minimal stand-in: exercise the wait/retry/mark sequence the manager uses."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    tracker = RateLimitTracker(cooldown_s=30.0)
    retried: set[tuple[str, str | None]] = set()
    chain: list[tuple[str, str | None]] = [("gemini", "m1")]
    msg = "HTTP 429 RESOURCE_EXHAUSTED retryDelay: '7s'"
    events: list[str] = []

    idx = 0
    while idx < len(chain):
        prov, model = chain[idx]
        wait_retry, delay_s = plan_rate_limit_retry(
            (prov, model), retried, msg, default_delay=tracker.cooldown_s
        )
        if wait_retry:
            events.append("wait")
            await asyncio.sleep(delay_s)
            chain.insert(idx + 1, (prov, model))
            idx += 1
            continue
        events.append("mark")
        tracker.mark_rate_limited(prov, model, cooldown_s=delay_s)
        break

    assert sleeps == [7.0]
    assert events == ["wait", "mark"]
    assert tracker.is_available("gemini", "m1") is False


class _FakeRateLimitedProvider:
    """Raises a Google-style 429 with retryDelay, then fails again on retry."""

    def __init__(self) -> None:
        self.calls = 0
        self.msg = (
            "HTTP 429 RESOURCE_EXHAUSTED. "
            "{'error': {'code': 429, 'details': ["
            "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
            "\"retryDelay\": \"11s\"}]}}"
        )

    async def complete(self, *_a, **_k):
        self.calls += 1
        raise RuntimeError(self.msg)


@pytest.mark.asyncio
async def test_fake_provider_429_reads_retry_delay_waits_retries_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fake provider: parse retryDelay, wait once, repeat request, then fallback."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    provider = _FakeRateLimitedProvider()
    tracker = RateLimitTracker(cooldown_s=30.0)
    retried: set[tuple[str, str | None]] = set()
    chain: list[tuple[str, str | None]] = [("gemini", "flash"), ("openai", "gpt")]
    events: list[str] = []
    fallback_hit: list[str] = []

    idx = 0
    while idx < len(chain):
        prov, model = chain[idx]
        try:
            await provider.complete()
            events.append("ok")
            break
        except RuntimeError as exc:
            msg = str(exc)
            wait_retry, delay_s = plan_rate_limit_retry(
                (prov, model), retried, msg, default_delay=tracker.cooldown_s
            )
            if wait_retry:
                events.append(f"wait:{delay_s}")
                await asyncio.sleep(delay_s)
                chain.insert(idx + 1, (prov, model))
                idx += 1
                continue
            events.append("mark")
            tracker.mark_rate_limited(prov, model, cooldown_s=delay_s)
            fallback_hit.append(chain[idx + 1][0] if idx + 1 < len(chain) else "")
            break

    assert provider.calls == 2  # first attempt + one retry
    assert sleeps == [11.0]
    assert events == ["wait:11.0", "mark"]
    assert fallback_hit == ["openai"]
    assert tracker.is_available("gemini", "flash") is False
