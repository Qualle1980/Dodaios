"""What a delegated order asks for: an answer about the code, or a change to it.

The mission prompt used to carry one standing directive for every order:
"deliver a complete, polished, production-quality result". For a change that
is the right floor. For a read order it is wrong: live 2026-10-23, a German
order to list the TODO comments in jarvis/missions/budget.py, read only,
produced a dump of the whole file (it has no TODOs), and the critic approved
it as a "complete deliverable".

Three kinds, decided from the user's own words only, never from the worker:

* ``read``    - list, find, show, count: the answer is a set of findings.
* ``analyze`` - explain, review, compare, why: the answer is an assessment.
* ``change``  - anything that creates, edits, deletes, runs or publishes.

A change verb always wins over a read verb ("find the bug and fix it" is a
change). An explicit read-only marker ("read only", "do not modify" and the
German equivalents below) makes the order read/analyze even when a change verb
appears in the negated phrase itself. Anything unrecognised is ``change``, so
an unclear order keeps the existing quality floor.
"""
from __future__ import annotations

import re
from typing import Final, Literal

TaskKind = Literal["read", "analyze", "change"]

# Explicit read-only markers. German entries are runtime-input classifiers.
_READ_ONLY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"read[\s-]?only|only\s+read|just\s+read|"
    r"(?:do\s+not|don'?t|never|without)\s+(?:chang|modif|edit|touch|writ)\w*|"
    r"no\s+changes|"
    r"nur\s+lesen|nur\s+anschauen|"  # i18n-allow: DE runtime-input classifier
    r"nichts\s+(?:ändern|aendern|verändern|veraendern|anfassen|schreiben)|"  # i18n-allow
    r"ohne\s+(?:etwas\s+)?(?:zu\s+)?(?:ändern|aendern|verändern)|"  # i18n-allow
    r"keine\s+änderungen|keine\s+aenderungen"  # i18n-allow
    r")\b",
    re.IGNORECASE,
)

_CHANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"fix|repair|change|modify|edit|update|rewrite|refactor|rename|move|delete|"
    r"remove|add|create|write|build|implement|generate|install|uninstall|"
    r"commit|push|merge|deploy|publish|release|run|execute|migrate|format|"
    r"send|upload|replace|make|"
    r"mach|mache|macht|behebe?|repariere?|ändere?|aendere?|bearbeite?|aktualisiere?|"  # i18n-allow
    r"schreibe?|erstelle?|baue?|implementiere?|generiere?|installiere?|"  # i18n-allow
    r"lösche?|loesche?|entferne?|füge|fuege|benenne|verschiebe?|"  # i18n-allow
    r"committe?|pushe?|starte?|führe|fuehre|ersetze?|formatiere?|sende?"  # i18n-allow
    r")\b",
    re.IGNORECASE,
)

_ANALYZE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"explain|analy[sz]e|review|assess|evaluate|compare|why|how\s+does|"
    r"summari[sz]e|audit|research\w*|recherch\w*|"  # i18n-allow
    r"erkläre?|erklaere?|analysiere?|prüfe?|pruefe?|bewerte?|vergleiche?|"  # i18n-allow
    r"warum|wieso|weshalb|fasse|untersuche?"  # i18n-allow
    r")\b",
    re.IGNORECASE,
)

_READ_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"list|find|show|count|search|locate|grep|which|where|what|"
    r"auflisten|liste|listen|finde?|zeige?|zähle?|zaehle?|suche?|"  # i18n-allow
    r"welche[rsnm]?|wo|was"  # i18n-allow
    r")\b",
    re.IGNORECASE,
)


def task_kind(request: str) -> TaskKind:
    """Classify the user's own request; see the module docstring."""
    text = (request or "").strip()
    if not text:
        return "change"
    read_only = bool(_READ_ONLY_RE.search(text))
    if not read_only and _CHANGE_RE.search(_READ_ONLY_RE.sub(" ", text)):
        return "change"
    if _ANALYZE_RE.search(text):
        return "analyze"
    if read_only or _READ_RE.search(text):
        return "read"
    return "change"


def is_read_task(request: str) -> bool:
    """True when the order must not touch files and is judged on its answer.

    A ``read`` order always is. An ``analyze`` order is only when the user
    said so ("nur lesen", "do not modify"): an analysis or research order
    without that marker may legitimately deliver a report file, so it keeps
    the existing quality floor and critic path.
    """
    kind = task_kind(request)
    if kind == "read":
        return True
    return kind == "analyze" and bool(_READ_ONLY_RE.search(request or ""))


__all__ = ["TaskKind", "is_read_task", "task_kind"]
