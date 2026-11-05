"""Brain health checker for the bootstrap sequence (Phase 5, CL-4).

Probes a brain provider with a minimal "hi" call and measures the time until
the first BrainDelta. Uses the Brain protocol (`async complete`) so that every
entry-point-registered provider (Claude, Gemini, OpenAI, …) can be validated
identically.

No auto-downgrade: the Q1 decision in the plan (§9) requires that aggressive
models such as `gemini-3.1-pro` / `gpt-5.5` are only used when they actually
exist. The caller (usually `factory._phase2_full_brain`) decides what to do
with a negative probe result — typically a bootstrap abort with an explicit
message. This class only returns the measurement data.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.core.protocols import BrainMessage, BrainRequest

if TYPE_CHECKING:
    from jarvis.brain.provider_registry import BrainProviderRegistry
    from jarvis.core.config import BrainTierConfig


class BrainConfigError(Exception):
    """Error in the brain-tier configuration (missing key, bad model, …)."""


@dataclass(frozen=True, slots=True)
class HealthResult:
    """Result of a single probe against a provider and model."""
    provider: str
    model: str
    ok: bool
    error: str | None = None
    duration_ms: float = 0.0


_PROBE_REQUEST = BrainRequest(
    messages=(BrainMessage(role="user", content="hi"),),
    max_tokens=8,
    # A connectivity probe does not need deterministic sampling. Some current
    # reasoning/tool models accept only their default temperature of 1, so
    # forcing 0 here turns a healthy provider into a false integration error.
    temperature=1.0,
    stream=True,
)


class BrainHealthChecker:
    """Validates brain providers via a minimal live call."""

    def __init__(self, registry: BrainProviderRegistry) -> None:
        self._registry = registry

    async def probe(
        self,
        provider: str,
        model: str,
        *,
        timeout_s: float = 5.0,
    ) -> HealthResult:
        """Calls the provider with a 1-token prompt and measures latency."""
        start = time.perf_counter()
        try:
            # Off the loop. Instantiating means the registry's entry-point
            # discovery — which IMPORTS every brain plugin the first time — and
            # the plugin's constructor. On a cold disk that import ran for
            # 15 s ON the event loop while a health sweep probed the providers
            # at boot (2026-08-26 and 2026-08-27, BUG-189), and every WebSocket
            # frame, HTTP route and brain turn stood still behind it.
            brain = await asyncio.to_thread(self._registry.instantiate, provider, model=model)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000.0
            return HealthResult(
                provider=provider,
                model=model,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )

        try:
            answered = await asyncio.wait_for(
                self._drain_first_delta(brain),
                timeout=timeout_s,
            )
        except TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return HealthResult(
                provider=provider,
                model=model,
                ok=False,
                error=f"timeout after {timeout_s:.1f}s",
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000.0
            return HealthResult(
                provider=provider,
                model=model,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        if not answered:
            # A stream that ended without ever carrying text or a tool call is
            # not a working provider. This used to read as success: the drain
            # broke on the first delta, so zero deltas raised nothing and fell
            # through to ok=True. The browser bridge answered "ok" that way on
            # 2026-10-30 while losing every real turn, and the status badge
            # repeated it. Silence is now reported as what it is.
            return HealthResult(
                provider=provider,
                model=model,
                ok=False,
                error="the provider closed the stream without answering",
                duration_ms=duration_ms,
            )
        return HealthResult(
            provider=provider,
            model=model,
            ok=True,
            error=None,
            duration_ms=duration_ms,
        )

    async def probe_tier(self, tier_config: BrainTierConfig) -> list[HealthResult]:
        """Probes the primary provider and all configured fallbacks sequentially."""
        pairs: list[tuple[str, str]] = [(tier_config.provider, tier_config.model)]
        if tier_config.fallback_provider and tier_config.fallback_model:
            pairs.append((tier_config.fallback_provider, tier_config.fallback_model))
        if tier_config.fallback_provider_2 and tier_config.fallback_model_2:
            pairs.append((tier_config.fallback_provider_2, tier_config.fallback_model_2))

        results: list[HealthResult] = []
        for provider, model in pairs:
            results.append(await self.probe(provider, model))
        return results

    @staticmethod
    async def _drain_first_delta(brain: object) -> bool:
        """Whether the stream carried an ANSWER; stops at the first one.

        Returns ``False`` for a stream that ends without ever producing text or
        a tool call — silence, or nothing but empty acknowledgement chunks.

        It drains to the first delta that carries something rather than the
        first delta of any kind. Several providers open with an empty role or
        keep-alive chunk before any text, so "first delta" and "first answer"
        are not the same event, and only the second one proves a turn ran.
        """
        stream = brain.complete(_PROBE_REQUEST)  # type: ignore[attr-defined]
        async for delta in stream:
            if getattr(delta, "content", None) or getattr(delta, "tool_call", None):
                return True
        return False
