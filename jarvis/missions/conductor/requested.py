"""The executor the user named themselves, when they named one.

Live 2026-10-25 the German order "Lass Codex nachsehen, wie viele Zeilen
jarvis/missions/conductor/loop.py hat." was conducted with the full team, so
the conductor kept picking ``conductor`` — itself, on the seat, with no file
access. It could not read the file, no round made progress, and the order
stalled out after two replans without ever reaching Codex.

A named worker is an instruction, not a hint. When the user's own words
delegate to one, that worker becomes the only executor for the order, which
also takes ``conductor`` out of the running as a substitute for it.

Only an explicit imperative counts. Merely mentioning an agent
("Warum funktioniert Codex nicht?") is a question about it, not an  # i18n-allow
order to it, and leaves the routing exactly as it was.
"""
from __future__ import annotations

import re
from typing import Final

#: Conductor executor name -> how the user may write it.
_AGENTS: Final[dict[str, str]] = {
    "codex": r"codex",
    "claude-code": r"claude[\s-]?code|claude[\s-]?cli",
}

#: Verbs that hand work over. German entries classify runtime input.
_CUE: Final[str] = (
    r"let|have|ask|tell|delegate|get|use|instruct|"
    r"lass|lasse|beauftrage?|frag|frage|delegiere|"  # i18n-allow
    r"übergib|uebergib|gib|schicke?|sende?"  # i18n-allow
)

#: ``<cue> [up to four words] <agent>`` — "Lass Codex ...", "gib das als
#: Auftrag an Codex", "ask codex to ...".
_BEFORE: Final[dict[str, re.Pattern[str]]] = {
    name: re.compile(rf"\b(?:{_CUE})\b(?:\s+\S+){{0,4}}?\s+(?:{alias})\b", re.IGNORECASE)
    for name, alias in _AGENTS.items()
}

#: ``<agent> soll ...`` — the same instruction, agent first.  # i18n-allow
_AFTER: Final[dict[str, re.Pattern[str]]] = {
    name: re.compile(rf"\b(?:{alias})\s+(?:soll(?:te)?|should|must)\b",  # i18n-allow
                     re.IGNORECASE)
    for name, alias in _AGENTS.items()
}


def requested_executor(request: str) -> str | None:
    """The executor the user delegated to, or ``None`` for ordinary routing.

    Returns a key of :data:`jarvis.missions.conductor.shadow.TEAM`. When the
    words name two agents, nobody can tell which the user meant, so this
    returns ``None`` and the conductor routes as usual.
    """
    text = (request or "").strip()
    if not text:
        return None
    named = [
        name for name in _AGENTS
        if _BEFORE[name].search(text) or _AFTER[name].search(text)
    ]
    return named[0] if len(named) == 1 else None


__all__ = ["requested_executor"]
