"""At most one model retry after a coding hand-off refusal, then stop.

Instant (and other owners) must not open a third coding pane/thread for the
same review after two model refusals. A refused model id is never retried with
the same id. Applies to Codex, Claude, and Grok coding hand-offs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Final

#: First open + one retry with a different confirmed id.
MAX_OPENS_PER_REVIEW: Final[int] = 2
MAX_REFUSALS_BEFORE_STOP: Final[int] = 2


@dataclass
class _ReviewAttempts:
    opens: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    updated_at: float = 0.0


class HandOffRetryGate:
    """Process-local gate keyed by owner review identity."""

    def __init__(self, *, ttl_s: float = 30 * 60.0) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._by_key: dict[str, _ReviewAttempts] = {}

    def _purge(self, now: float) -> None:
        dead = [k for k, v in self._by_key.items() if now - v.updated_at > self._ttl_s]
        for key in dead:
            self._by_key.pop(key, None)

    def allow_open(self, key: str, model_id: str) -> str | None:
        """Return ``None`` when the open may proceed, else an honest stop reason."""
        wanted = (model_id or "").strip()
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            state = self._by_key.get(key)
            if state is None:
                return None
            if wanted and wanted in state.refused:
                return (
                    f"review not run: model {wanted!r} was already refused for this "
                    "review; choose a different confirmed id or stop."
                )
            if len(state.refused) >= MAX_REFUSALS_BEFORE_STOP:
                reason = state.refused[-1] if state.refused else "model refused"
                return (
                    f"review not run: coding hand-off already refused twice "
                    f"(last={reason!r}); no further pane or turn is opened."
                )
            if len(state.opens) >= MAX_OPENS_PER_REVIEW:
                return (
                    "review not run: coding hand-off already opened the maximum "
                    f"of {MAX_OPENS_PER_REVIEW} panes for this review."
                )
            return None

    def note_open(self, key: str, model_id: str) -> None:
        """Record that a pane/thread was opened for ``key`` with ``model_id``."""
        wanted = (model_id or "").strip() or "(default)"
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            state = self._by_key.setdefault(key, _ReviewAttempts())
            state.opens.append(wanted)
            state.updated_at = now

    def note_refusal(self, key: str, model_id: str, *, reason: str = "") -> None:
        """Record a model refusal for ``key`` (confirmation error, not_accepted, …)."""
        wanted = (model_id or "").strip() or "(default)"
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            state = self._by_key.setdefault(key, _ReviewAttempts())
            if wanted not in state.refused:
                state.refused.append(wanted)
            state.updated_at = now
            _ = reason  # callers pass it for logs; the gate keys on model ids only

    def reset(self, key: str) -> None:
        with self._lock:
            self._by_key.pop(key, None)

    def clear_all(self) -> None:
        """Tests only."""
        with self._lock:
            self._by_key.clear()


_SHARED = HandOffRetryGate()


def shared_gate() -> HandOffRetryGate:
    return _SHARED


def review_key(*, owner_session: str = "", fingerprint: str = "", title: str = "") -> str:
    """Stable key for one coding review / hand-off."""
    parts = [p.strip() for p in (owner_session, fingerprint, title) if p and p.strip()]
    return "|".join(parts) or "anonymous-review"


__all__ = [
    "HandOffRetryGate",
    "MAX_OPENS_PER_REVIEW",
    "MAX_REFUSALS_BEFORE_STOP",
    "review_key",
    "shared_gate",
]
