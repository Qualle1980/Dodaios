"""Does this turn need the user's people, contacts and remembered facts?

Those blocks are the expensive half of the system prompt: the contact index
alone is rendered with an 8_000-character cap, and people, ambient core memory
and the learning loop's observations sit beside it. Measured live 2026-10-21,
"Was ist 10 plus 10?" carried 12_021 input tokens — the personal corpus went
out with a question that could not use a single line of it.

The gate is deterministic and errs toward INCLUDING the context: a turn that
needed a name and did not get it answers wrongly, while a turn that carried it
needlessly only costs tokens. So this returns False only for utterances that
are provably impersonal — a clock or date question, arithmetic, a greeting, an
acknowledgement — and True for everything else, including anything it cannot
classify.
"""
from __future__ import annotations

import re

#: STRONG personal markers: these decide on their own, and they outrank an
#: impersonal shape — "wie spät ist es bei meiner Schwester" is a clock
#: question that still needs the people index. Weak pronouns are deliberately
#: NOT here: "was haben wir heute für einen Tag" says "wir" and means nobody.
_PERSONAL_STRONG = re.compile(
    r"\b("
    r"mein\w*|unser\w*|"  # i18n-allow
    r"my|mine|our|ours|"
    r"wer|wem|wen|jemand|niemand|person|leute|freund\w*|kollege\w*|kollegin\w*|"  # i18n-allow
    r"familie|schwester|bruder|mutter|vater|frau|mann|kind\w*|chef\w*|kunde\w*|"  # i18n-allow
    r"who|whom|someone|anyone|people|friend\w*|colleague\w*|family|sister|brother|"
    r"mother|father|client\w*|"
    r"kontakt\w*|telefon\w*|nummer|handy|adresse|e-?mail|"  # i18n-allow
    r"contact\w*|phone|number|address|"
    r"erinner\w*|ged[äa]chtnis|gemerkt|notiert|wei[ßs]t\s+du\s+noch|"  # i18n-allow
    r"remember|recall|memory|noted|"
    r"hei[ßs]\w*|name|nennt"  # i18n-allow
    r")\b",
    re.IGNORECASE,
)

#: Shapes whose answer cannot involve the personal corpus. Matched anywhere in
#: the utterance, because "welches Datum ist heute" carries the shape plus
#: filler, and an anchored match would miss every natural phrasing.
_IMPERSONAL = (
    # clock / date / weekday
    r"wie\s+viel\s+uhr|wie\s+sp[äa]t|welcher\s+tag|welchen\s+tag|"  # i18n-allow
    # "für einen Tag" unanchored: the live utterance was "was haben wir heute
    # für einen Tag", so requiring "was" directly before it missed the real
    # phrasing (measured 2026-10-21).
    r"f[üu]r\s+ein(?:en)?\s+tag|welches\s+datum|der\s+wievielte|"  # i18n-allow
    r"was\s+ist\s+heute|welcher\s+wochentag|"  # i18n-allow
    r"what\s+time|what\s+day|what(?:'s|\s+is)\s+the\s+date|"
    # arithmetic
    r"\d+\s*(?:plus|minus|mal|geteilt\s+durch|durch|[+\-*/x×÷])\s*\d+|"  # i18n-allow
    # closing noise
    r"^\s*(?:danke\w*|vielen\s+dank|tsch[üu]ss|ciao|bis\s+sp[äa]ter|ok(?:ay)?|"  # i18n-allow
    r"passt|verstanden|thanks|thank\s+you|bye|goodbye|got\s+it|alright)\s*$"
)
_IMPERSONAL_RE = re.compile(_IMPERSONAL, re.IGNORECASE)


def wants_personal_context(user_text: str) -> bool:
    """True unless the utterance provably cannot use the personal corpus.

    ``True`` is the default and the safe answer. ``False`` needs an impersonal
    shape AND no strong personal marker, so a clock question about a relative
    keeps the people index while a bare clock question does not.
    """
    text = (user_text or "").strip()
    if not text:
        return True
    core = re.sub(r"^(?:hallo|hey|hi|jarvis)[,\s]+", "", text, flags=re.IGNORECASE)
    core = core.strip().strip("?!.,").strip()
    if not core:
        return True
    if _PERSONAL_STRONG.search(core):
        return True
    return not _IMPERSONAL_RE.search(core)


__all__ = ["wants_personal_context"]
