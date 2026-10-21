"""ContinuationWindow — re-attach a fast-follow utterance to the in-flight turn.

Sibling of :mod:`jarvis.speech.continuation_buffer`. Where ContinuationBuffer
coalesces a *syntactically* open fragment BEFORE dispatch, ContinuationWindow
covers the case the maintainer reported (2026-06-16): the user keeps talking
while the brain is ALREADY thinking/speaking. The pipeline aborts the
half-formed answer (Unit B) and, on the next utterance, prepends the
just-dispatched text so the whole sentence is re-thought as ONE turn.

Stdlib-only, deterministic (clock injected), fail-open. The window holds:

* ``text``        — the last dispatched user text, eligible to be extended.
* ``chain``       — fragments coalesced into the current window (bounded).
* ``deadline_ns`` — ``None`` while the arming turn is still in flight (always
                    active); a wall-clock deadline once the turn went idle
                    (grace countdown). Expiry is checked lazily on the next
                    ``try_recombine`` — never via a background timer that could
                    fire across turns (BUG-032 watchdog-class avoidance).

Design contract: ``try_recombine`` is NON-destructive on success (it leaves the
window armed with the prior text); the pipeline overwrites the text via
``note_dispatch`` only once it actually commits the combined turn to the brain.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

_DEFAULT_GRACE_MS: Final[int] = 2500
_DEFAULT_MAX_CHAIN: Final[int] = 3

#: Hard ceiling on how long a freeze (``note_speech_resumed``) may keep the
#: window open past the arming turn. A resumed utterance used to clear the
#: deadline outright, which armed the window FOREVER — and the microphone hears
#: the assistant's own speech tail, so that freeze fired after every single
#: answer. Measured live 2026-10-21: a VAD "speech start" landed 0.4-1.3 s after
#: each reply at rms 0.003-0.034 (near-silence, not the user), and the next real
#: question — seconds later — was glued onto the previous one. Three questions
#: in a row were answered together, and the local zero-token answers were
#: bypassed because the combined text matched no one-shot pattern.
#:
#: A spoken continuation finalizes within a few seconds of starting; anything
#: longer is a new request. This bounds the freeze without shortening a genuine
#: continuation.
_DEFAULT_FREEZE_CEILING_MS: Final[int] = 5000


#: Openers that start a question or a command outright. A continuation
#: fragment does not begin this way — it begins mid-thought, with a
#: conjunction, a correction or a bare noun phrase.
_REQUEST_OPENERS: Final[tuple[str, ...]] = (
    # DE question words and polite command openers
    "wie", "was", "wer", "wen", "wem", "wann", "wo", "warum", "wieso",
    "weshalb", "welche", "welcher", "welches", "welchen", "wieviel",
    "kannst", "kannst du", "könntest", "koenntest", "zeig", "zeige",
    "sag", "sage", "erzähl", "erzaehl", "erkläre", "erklaere", "mach",
    "öffne", "oeffne", "starte", "schreib", "schreibe", "such", "suche",
    "rechne", "gib", "nenne", "bitte", "hallo", "jarvis", "danke",
    # EN
    "how", "what", "who", "whom", "when", "where", "why", "which",
    "can", "could", "would", "please", "show", "tell", "explain", "open",
    "start", "write", "search", "give", "hello", "thanks",
    # AR
    "كم", "ما", "ماذا", "من", "كيف", "متى", "وين", "ليش", "لماذا", "أي",
)


def is_self_contained_request(text: str) -> bool:
    """True when ``text`` reads as a complete new request, not a continuation.

    Deliberately conservative: it must be WRONG as rarely as possible in the
    direction that splits a real continuation, because that only costs an
    extra turn, while a false merge produces a wrong answer to two questions
    at once. So it says "self-contained" only on a clear signal — the
    utterance opens with a question word or an imperative, or it ends in a
    question mark.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith("?") or stripped.endswith("؟"):
        return True
    lowered = stripped.lower().lstrip("\"'“„«(-– ")
    # Trailing punctuation is part of the utterance, not of the word: without
    # stripping it "Danke." never matched the opener "danke".
    words = [w.strip(".,!?;:…\"'»«)") for w in lowered.split()]
    words = [w for w in words if w]
    if not words:
        return False
    first_two = " ".join(words[:2])
    return any(
        candidate in _REQUEST_OPENERS for candidate in (first_two, words[0])
    )


class ContinuationWindow:
    """Tracks the last dispatched utterance so a continuation can re-attach."""

    def __init__(
        self,
        *,
        grace_ms: int = _DEFAULT_GRACE_MS,
        max_chain: int = _DEFAULT_MAX_CHAIN,
        freeze_ceiling_ms: int = _DEFAULT_FREEZE_CEILING_MS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if grace_ms < 0:
            raise ValueError("grace_ms must be >= 0")
        if max_chain < 1:
            raise ValueError("max_chain must be >= 1")
        if freeze_ceiling_ms < 0:
            raise ValueError("freeze_ceiling_ms must be >= 0")
        self._grace_ns = int(grace_ms) * 1_000_000
        self._freeze_ceiling_ns = int(freeze_ceiling_ms) * 1_000_000
        self._max_chain = int(max_chain)
        self._clock = clock or time.monotonic_ns
        self._text: str = ""
        self._chain: int = 0
        self._deadline_ns: int | None = None
        #: Absolute cutoff for a frozen window, set when the arming turn goes
        #: idle. ``note_speech_resumed`` may push the deadline out but never
        #: past this, so a spurious VAD trigger cannot arm the window forever.
        self._freeze_cutoff_ns: int | None = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def is_armed(self) -> bool:
        return self._chain > 0

    def note_dispatch(self, text: str, *, continued: bool) -> None:
        """Record a turn that just committed to the brain.

        ``continued`` marks whether THIS dispatch was itself a recombine
        (chain grows) or a fresh turn (chain resets to 1). The window becomes
        'in flight' (no deadline) until ``mark_idle`` starts the grace countdown.
        """
        self._text = text.strip()
        self._chain = (self._chain + 1) if continued else 1
        self._deadline_ns = None
        self._freeze_cutoff_ns = None

    def mark_idle(self) -> None:
        """The armed turn finished (answer spoken or aborted): start grace."""
        if self.is_armed:
            now = self._clock()
            self._deadline_ns = now + self._grace_ns
            # From here a freeze may extend the deadline, but only this far.
            self._freeze_cutoff_ns = now + self._freeze_ceiling_ns

    def note_speech_resumed(self) -> None:
        """The user started speaking again — freeze the grace countdown.

        The grace started at turn end (``mark_idle``) measures THINKING silence.
        Once the user is actually forming the follow-up, the clock must not keep
        running against them: a slow-to-finalize continuation would otherwise
        miss ``try_recombine`` even though it began well inside the grace (live
        bug 2026-06-18, session 71f2d2de: ~3 s to formulate the next fragment >
        the 2.5 s grace, so it became a fresh turn). Re-enters the 'in flight'
        state (deadline cleared) — but ONLY while still armed and not already
        expired, so a genuinely late resume cannot resurrect a dead window.
        Fail-open / idempotent.
        """
        if not self.is_armed:
            return
        now = self._clock()
        if self._deadline_ns is not None and now > self._deadline_ns:
            return
        if self._freeze_cutoff_ns is not None and now > self._freeze_cutoff_ns:
            # The freeze has already been extended as far as it may go. Leave
            # the expired deadline in place: a VAD trigger this late is noise
            # or a new request, not the continuation of a finished turn.
            return
        # Extend to the hard ceiling -- once, and no further. The user needs
        # room to formulate a follow-up (~3 s measured, session 71f2d2de),
        # which is why this does more than restart the grace; clearing the
        # deadline OUTRIGHT is what armed the window indefinitely, and the
        # microphone hears the assistant's own speech tail, so that fired
        # after every single answer (see _DEFAULT_FREEZE_CEILING_MS).
        # Naturally idempotent: a second trigger cannot push it past the cap.
        if self._freeze_cutoff_ns is None:
            self._deadline_ns = now + self._grace_ns
        else:
            self._deadline_ns = self._freeze_cutoff_ns

    def is_live(self) -> bool:
        """True if the NEXT utterance WOULD recombine — armed, not expired, and
        under the chain cap. NON-mutating mirror of ``try_recombine``'s gate, so
        a caller (e.g. the pipeline tagging ``TranscriptFinal.continues_previous``
        before dispatch) can learn the recombine decision without consuming the
        window. Fail-safe: never raises."""
        if not self.is_armed:
            return False
        if self._deadline_ns is not None and self._clock() > self._deadline_ns:
            return False
        if self._chain >= self._max_chain:
            return False
        return True

    def try_recombine(self, new_text: str) -> str | None:
        """Return ``prior + new`` if a continuation is live, else ``None``.

        Non-destructive on success (window stays armed with the prior text).
        Expired or over-cap -> clears and returns ``None`` (fresh turn).
        """
        if not self.is_armed:
            return None
        if self._deadline_ns is not None and self._clock() > self._deadline_ns:
            self.clear()
            return None
        if self._chain >= self._max_chain:
            self.clear()
            return None
        if is_self_contained_request(new_text):
            # A whole new question is not the tail of the previous one. Gluing
            # them made the model answer BOTH, and the answers piled up across
            # turns: measured live 2026-10-21, "Wie viel ist 10 plus 10?" was
            # sent as "wie viel ist 100 plus 100? كم هو 2 مقارنة مع 2؟ Wie viel
            # ist 10 plus 10?" and came back as all three answers in one
            # breath. It also bypassed the local zero-token answers, because
            # the combined text matches no one-shot pattern.
            #
            # The feature's own examples are FRAGMENTS ("…nicht australische",
            # "sondern der weiteste Ort"), which this still merges.
            self.clear()
            return None
        return f"{self._text} {new_text.strip()}".strip()

    def clear(self) -> None:
        self._text = ""
        self._chain = 0
        self._deadline_ns = None
        self._freeze_cutoff_ns = None


__all__ = ["ContinuationWindow"]
