"""Flash-shaped adapter over a connected subscription Brain (chat seat).

When ``brain.primary`` is a plan seat the Flash REGISTRY has no adapter for,
spawn announcements and readbacks still need a live composer. This wraps
``resolve_subscription_brain(..., prefer="chatgpt-plan")`` behind the same
``AbstractAckProvider.run`` surface the Spawn-Announcer and Readback-Composer
already call — never Ollama, never a second family's API key.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class SubscriptionBrainFlashAck:
    """Single-shot flash adapter that spends a connected subscription Brain."""

    def __init__(
        self,
        brain: Any,
        *,
        provider_name: str = "chatgpt-plan",
        temperature: float = 0.6,
        max_output_tokens: int = 40,
    ) -> None:
        self._brain = brain
        self.provider_name = provider_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    async def run(
        self,
        utterance: str,
        language: str,
        *,
        persona_prompt: str,
    ) -> str | None:
        del language  # the persona prompt already carries the turn language
        try:
            from jarvis.core.protocols import BrainMessage, BrainRequest

            req = BrainRequest(
                messages=(BrainMessage(role="user", content=utterance),),
                system=persona_prompt,
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                stream=False,
                reasoning_effort="none",
            )
            parts: list[str] = []
            async for delta in self._brain.complete(req):
                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
            joined = "".join(parts).strip()
            return joined or None
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            log.warning(
                "Subscription flash ack on %s failed: %s",
                self.provider_name,
                exc,
            )
            return None
