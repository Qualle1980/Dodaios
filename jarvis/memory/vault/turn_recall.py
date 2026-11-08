"""Memories that match THIS message, for the turn context (ADR-0044, chunk 3).

The standing block answers "who is this person?". This answers the other
question: "did they tell me something about what they just asked?" — and it is
the half the standing block cannot do, because a budget that always fits
cannot hold everything someone ever said.

Cheap and local by design: one FTS query, no model call, so it is safe where a
user is waiting (AP-9, AP-11). Two rules make it worth its tokens:

- **Never repeat the standing block.** A memory already in the always-on card
  would otherwise be paid for twice in the same prompt.
- **A relative floor.** Keyword search always returns something; only hits
  close to the best one are worth a line, or a question about the weather
  drags in whatever shares a word with it.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

#: Most memories one message may add.
TURN_LIMIT = 6
#: Characters the block may use.
TURN_MAX_CHARS = 900
#: A hit must score at least this share of the best hit to be shown.
TURN_RELATIVE_FLOOR = 0.35
#: Shortest word that may carry a topic.
MIN_WORD_CHARS = 3

HEADER = (
    "## Possibly relevant memories (looked up for this message)\n"
    "From the user's long-term memory. Use one only if it actually fits the "
    "question; say nothing about the rest, and never mention that you looked."
)

#: Words that say nothing about what a message is about, in the languages the
#: product takes input in. Without them "what do you think about that?" queries
#: the store for "think" and "about".
#: Each word is casefolded the same way the message is, because German sharp s
#: casefolds to a double s: a literal stopword written with the one would never
#: match what a message produces, and the word would be treated as a topic.
#: The folded forms below are speech-input vocabulary.
_STOPWORDS = frozenset(
    word.casefold()
    for word in (
        "the and you your what when where who how why can could would should will "
        "are was were have has had this that these those with from for about into "
        "please tell know remember just also some any more much very did does "
        "der die das und ich du wir ihr sie ist sind war wer wie wann warum "  # i18n-allow
        "ein eine einen dem den des mit von für über auf noch auch bitte mal mir "  # i18n-allow
        "mich dir dich mein meine kannst kann weißt hast habe nicht oder aber "  # i18n-allow
        "que los las una por para como con del pero"  # i18n-allow
    ).split()
)

_WORD = re.compile(r"\w+", re.UNICODE)


def content_words(user_text: str) -> list[str]:
    """The words of a message that could name a topic, in order, deduplicated."""
    seen: dict[str, None] = {}
    for raw in _WORD.findall(user_text.casefold()):
        if len(raw) >= MIN_WORD_CHARS and raw not in _STOPWORDS:
            seen.setdefault(raw, None)
    return list(seen)


def _line(memory: Any) -> str:
    line = f"- {memory.text}"
    # An imported memory's start date is the import, not when it became true.
    if memory.valid_from and memory.origin != "imported":
        line += f" (since {memory.valid_from[:10]})"
    return line


def turn_block(
    vault: Any,
    user_text: str,
    *,
    standing_ids: frozenset[str] = frozenset(),
    limit: int = TURN_LIMIT,
    max_chars: int = TURN_MAX_CHARS,
) -> tuple[str, list[str]]:
    """The block for this message and the ids it shows.

    Returns ``("", [])`` when the message has no content words, nothing
    matches, or everything that matched is already in the standing block. The
    caller counts the shown ids as surfaced — which is not the same as helpful
    (ADR-0044 principle 7).
    """
    words = content_words(user_text)
    if not words:
        return "", []
    try:
        hits = vault.search(" ".join(words), limit=max(limit * 2, 8))
    except Exception:  # noqa: BLE001 — recall must never break a turn
        log.warning("memory vault: per-message recall failed", exc_info=True)
        return "", []
    hits = [hit for hit in hits if hit.memory.id not in standing_ids]
    if not hits:
        return "", []

    floor = hits[0].score * TURN_RELATIVE_FLOOR
    lines: list[str] = []
    shown: list[str] = []
    used = 0
    for hit in hits:
        if hit.score < floor or len(lines) >= limit:
            break
        line = _line(hit.memory)
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        shown.append(hit.memory.id)
        used += len(line) + 1
    if not lines:
        return "", []
    return HEADER + "\n" + "\n".join(lines), shown
