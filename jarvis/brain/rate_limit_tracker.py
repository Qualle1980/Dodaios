"""RateLimitTracker: circuit breaker for rate-limited providers.

When a brain provider has returned a 429, we mark it as "unavailable" for
`cooldown_s` seconds — the fallback chain skips it. After the cooldown
expires, it is tried again.

This prevents the manager from wasting ~3-5 s per voice turn on overloaded
providers, only to fall back to the same fallback every time.

A first 429 on a (provider, model) in a turn waits the API ``retryDelay``
once and retries the same request; only a second 429 marks the cooldown and
falls through the chain (Phase-2 decision).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

#: Same shape as ``jarvis.plugins.tts.gemini_flash_tts._RETRY_DELAY_RE``.
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")

#: Upper bound for a single wait-before-retry (misparsed hour-long quotas).
_MAX_WAIT_S = 300.0


def parse_retry_delay(error_msg: str, *, default: float = 30.0) -> float:
    """Seconds to wait from a provider 429 body; ``default`` when absent."""
    m = _RETRY_DELAY_RE.search(error_msg or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float(default)


def clamp_retry_wait(delay_s: float) -> float:
    """Bound a wait to ``[0, _MAX_WAIT_S]`` so a bad parse cannot hang a turn."""
    try:
        value = float(delay_s)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    return min(value, _MAX_WAIT_S)


def plan_rate_limit_retry(
    key: tuple[str, str | None],
    retried: set[tuple[str, str | None]],
    error_msg: str,
    *,
    default_delay: float = 30.0,
) -> tuple[bool, float]:
    """Whether to wait-and-retry once, and which delay to use / mark.

    Returns ``(wait_and_retry, delay_s)``. First hit for ``key`` → wait+retry
    (caller must add ``key`` to ``retried``). Second hit → do not wait; caller
    marks cooldown with ``delay_s`` and falls back.
    """
    delay = clamp_retry_wait(parse_retry_delay(error_msg, default=default_delay))
    if key in retried:
        return False, delay
    retried.add(key)
    return True, delay


@dataclass
class RateLimitTracker:
    cooldown_s: float = 30.0
    _marks: dict[tuple[str, str | None], float] = field(default_factory=dict)

    def is_available(self, provider: str, model: str | None = None) -> bool:
        key = (provider, model)
        deadline = self._marks.get(key)
        if deadline is None:
            return True
        if time.time() >= deadline:
            # Cooldown expired — new attempt allowed
            self._marks.pop(key, None)
            return True
        return False

    def mark_rate_limited(self, provider: str, model: str | None = None,
                          cooldown_s: float | None = None) -> None:
        key = (provider, model)
        self._marks[key] = time.time() + (cooldown_s or self.cooldown_s)

    def clear(self, provider: str | None = None) -> None:
        if provider is None:
            self._marks.clear()
            return
        self._marks = {k: v for k, v in self._marks.items() if k[0] != provider}

    def snapshot(self) -> dict[str, float]:
        now = time.time()
        return {
            f"{p}:{m}" if m else p: max(0.0, deadline - now)
            for (p, m), deadline in self._marks.items()
        }
