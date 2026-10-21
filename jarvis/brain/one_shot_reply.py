"""Answer a closed-set one-shot without asking a model.

Measured live on 2026-10-20: "danke" took 7.8 s, because the turn walked the
whole provider chain and waited for a model answer. There is nothing for a
model to decide there — the intent router already recognises the utterance as a closed-set
one-shot, and a thank-you, a greeting, a farewell, an acknowledgement, the
time and the date all have one right answer that this process can give in
microseconds.

Deliberately narrow:

* only the kinds :func:`jarvis.brain.intent_router.one_shot_kind` recognises,
  which match the WHOLE utterance, so a thank-you followed by a real
  question is not one of them and still reaches the model;
* a GREETING is deliberately NOT answered here, although the router
  recognises one. A greeting usually opens an interaction, and that is where
  the assistant's own character and whatever it can see belong; the turns
  below are closing noise or a lookup, where a canned line loses nothing;
* only the languages a line exists for. An unknown reply language returns
  ``None`` and the turn proceeds normally, because a German answer to a
  Portuguese greeting is worse than a slow one;
* no state, no clock drift: the caller passes the time, so a test pins it.

The phrasing rotates deterministically on the utterance so two greetings in a
row do not read like a stuck recording.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
from zoneinfo import ZoneInfo

from .intent_router import one_shot_kind

#: Reply pools per kind and language. Short on purpose: these are spoken as
#: well as written, and a one-shot that triggers a paragraph is a worse answer
#: than a slow one.
_LINES: dict[str, dict[str, tuple[str, ...]]] = {
    "thanks": {
        "de": ("Gerne.", "Jederzeit.", "Immer gern."),  # i18n-allow: runtime reply
        "en": ("Anytime.", "Happy to help.", "Sure thing."),
        "es": ("De nada.", "Cuando quieras.", "Un placer."),  # i18n-allow
    },
    "ack": {
        "de": ("Alles klar.", "Passt.", "Verstanden."),  # i18n-allow: runtime reply
        "en": ("Got it.", "Understood.", "All right."),
        "es": ("Entendido.", "Vale.", "De acuerdo."),  # i18n-allow
    },
    "farewell": {
        "de": ("Bis später.", "Tschüss.", "Mach's gut."),  # i18n-allow: runtime reply
        "en": ("See you.", "Bye.", "Take care."),
        "es": ("Hasta luego.", "Adiós.", "Cuídate."),  # i18n-allow
    },
}

#: How the computed answers are phrased, per language.
_CLOCK: dict[str, str] = {
    "de": "Es ist {time} Uhr.",  # i18n-allow: runtime reply
    "en": "It is {time}.",
    "es": "Son las {time}.",  # i18n-allow
}
_DATE: dict[str, str] = {
    "de": "Heute ist {date}.",  # i18n-allow: runtime reply
    "en": "Today is {date}.",
    "es": "Hoy es {date}.",  # i18n-allow
}
#: The weekday leads: "was haben wir heute fuer einen Tag" asks for the day,
#: not the ordinal, and locale-independent names keep this off the OS locale
#: (``%A`` returns whatever the host is set to).
_WEEKDAYS: dict[str, tuple[str, ...]] = {
    "de": ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag",  # i18n-allow
           "Samstag", "Sonntag"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
           "Saturday", "Sunday"),
    "es": ("lunes", "martes", "miércoles", "jueves", "viernes",  # i18n-allow
           "sábado", "domingo"),
}
_DATE_FORMAT: dict[str, str] = {"de": "%d.%m.%Y", "en": "%B %d, %Y", "es": "%d/%m/%Y"}
_TIME_FORMAT: dict[str, str] = {"de": "%H:%M", "en": "%I:%M %p", "es": "%H:%M"}


#: Spoken arithmetic -> operator. Only two-operand sums land here.
_OPS: dict[str, str] = {
    "plus": "+", "+": "+",
    "minus": "-", "-": "-",
    "mal": "*", "times": "*", "*": "*", "x": "*", "×": "*",
    "geteilt durch": "/", "durch": "/", "divided by": "/", "/": "/", "÷": "/",
}
_MATH_RE = _re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*"
    r"(plus|minus|mal|times|geteilt\s+durch|durch|divided\s+by|[+\-*/x×÷])\s*"
    r"(-?\d+(?:[.,]\d+)?)",
    _re.IGNORECASE,
)
_MATH: dict[str, str] = {
    "de": "{a} {verb} {b} ist {r}.",  # i18n-allow: runtime reply
    "en": "{a} {verb} {b} is {r}.",
    "es": "{a} {verb} {b} es {r}.",  # i18n-allow
}
_MATH_VERB: dict[str, dict[str, str]] = {
    "de": {"+": "plus", "-": "minus", "*": "mal", "/": "geteilt durch"},  # i18n-allow
    "en": {"+": "plus", "-": "minus", "*": "times", "/": "divided by"},
    "es": {"+": "más", "-": "menos", "*": "por", "/": "entre"},  # i18n-allow
}


def _format_number(value: float) -> str:
    """A spoken number: no trailing ``.0``, and no float noise."""
    rounded = round(value, 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def _math_answer(text: str, language: str) -> str | None:
    """The computed answer for a two-operand sum, or None.

    Pure arithmetic on parsed numbers -- never ``eval``, so no utterance can
    execute anything. A division by zero returns None and the turn proceeds
    normally: a model can explain it, this cannot.
    """
    template = _MATH.get(language)
    verbs = _MATH_VERB.get(language)
    if template is None or verbs is None:
        return None
    m = _MATH_RE.search(text)
    if m is None:
        return None
    raw_op = " ".join(m.group(2).lower().split())
    op = _OPS.get(raw_op)
    if op is None:
        return None
    try:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(3).replace(",", "."))
    except ValueError:
        return None
    if op == "/" and b == 0:
        return None
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0.0}[op]
    return template.format(
        a=_format_number(a), verb=verbs[op], b=_format_number(b), r=_format_number(result)
    )


def _rotate(pool: tuple[str, ...], seed: str) -> str:
    """A stable pick: the same utterance answers the same way, two different
    ones usually do not. No RNG, so a test reads like the runtime."""
    return pool[sum(ord(ch) for ch in seed) % len(pool)]


def one_shot_reply(
    user_text: str,
    lang: str,
    *,
    now: _dt.datetime | None = None,
    tz: str | None = None,
) -> str | None:
    """The deterministic answer for ``user_text``, or None to use a model."""
    kind = one_shot_kind(user_text)
    if kind is None:
        return None
    language = (lang or "").split("-")[0].lower()

    if kind in _LINES:
        pool = _LINES[kind].get(language)
        return _rotate(pool, user_text.strip().lower()) if pool else None
    if kind == "math":
        return _math_answer(user_text, language)
    if kind not in ("clock", "date"):
        # A recognised kind with no line of its own — a greeting — belongs to
        # the model. Falling through to the computed branch below would have
        # answered "hallo" with today's date.
        return None

    template = (_CLOCK if kind == "clock" else _DATE).get(language)
    if template is None:
        return None
    moment = now or _dt.datetime.now(ZoneInfo(tz) if tz else None)
    if kind == "clock":
        return template.format(time=moment.strftime(_TIME_FORMAT[language]).lstrip("0"))
    weekday = _WEEKDAYS[language][moment.weekday()]
    return template.format(
        date=f"{weekday}, {moment.strftime(_DATE_FORMAT[language])}"
    )


__all__ = ["one_shot_reply"]
