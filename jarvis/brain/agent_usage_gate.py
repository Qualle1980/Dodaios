"""Explicit-request gate for the ``agent-usage`` tool.

Plan percentages belong only on a turn that asked for them. Offering the tool
on every turn invites the model to narrate remaining quota in ordinary
answers; hiding it unless the utterance asks keeps those numbers off the
wire until the user wants them.

Regex-only, no model call (AP-11). A false negative is cheap — the user asks
again with a clearer wording. A false positive would put usage figures into a
turn that never asked, which is the failure this gate exists to prevent.
"""

from __future__ import annotations

import re

# German and English input vocabulary for "how much of my plan is left/used".
_USAGE_NOUN = (
    r"(?:"
    r"wochenlimit|tageslimit|stundenlimit|nutzungslimit|kontingent|"  # i18n-allow: input vocab
    r"verbrauch|restkontingent|restprozent(?:e)?|nutzung|"  # i18n-allow: input vocab
    r"quota|usage|rate[\s-]?limit|plan[\s-]?limit|weekly[\s-]?limit|"
    r"session[\s-]?limit|usage[\s-]?limit"
    r")"
)
_PERCENT = r"(?:prozent(?:satz)?|percent(?:age)?|%)"  # i18n-allow: input vocab
_PROVIDER = (
    r"(?:"
    r"claude|anthropic|codex|chatgpt|openai|grok(?:[\s-]?build)?|"
    r"abo|abonnement|subscription|seat|konto"  # i18n-allow: input vocab
    r")"
)
_USED_OR_LEFT = (
    r"(?:"
    r"verbraucht|übrig|uebrig|rest|noch|"  # i18n-allow: input vocab
    r"used|spent|remaining|left|available"
    r")"
)

_DIRECT = re.compile(
    rf"\b{_USAGE_NOUN}\b",
    re.IGNORECASE,
)
_PERCENT_WITH_PROVIDER_OR_SPENT = re.compile(
    rf"\b{_PERCENT}\b.{{0,48}}\b(?:{_PROVIDER}|{_USED_OR_LEFT}|{_USAGE_NOUN})\b|"
    rf"\b(?:{_PROVIDER}|{_USED_OR_LEFT}|{_USAGE_NOUN})\b.{{0,48}}\b{_PERCENT}\b",
    re.IGNORECASE,
)


def wants_agent_usage(text: str) -> bool:
    """True when ``text`` asks for subscription plan usage or remaining percent."""
    utterance = (text or "").strip()
    if not utterance:
        return False
    if _DIRECT.search(utterance):
        return True
    return _PERCENT_WITH_PROVIDER_OR_SPENT.search(utterance) is not None


__all__ = ["wants_agent_usage"]
