"""Offline pin for D4's identity-card contribution to language breaks.

A live turn is still required to prove the model follows the output-language
pin. This test only shows the standing card no longer ships contradictory
language instructions — the plausible, unproven contributor from the
2026-11-14 live test.
"""

from __future__ import annotations

from jarvis.brain import identity_card as ic

_CONFLICTING = """# Pat Example

## Identity

- Primary language: Arabic

## Preferences

- Reply language: German
- Speaks Arabic, German and English
"""


def test_conflicting_language_lines_collapse_to_one_with_provenance() -> None:
    card = ic.distill_identity_card(profile_markdown=_CONFLICTING)
    language_lines = [
        line
        for line in card.splitlines()
        if any(
            token in line.casefold()
            for token in ("language", "speaks", "arabic", "german", "english")
        )
    ]
    assert len(language_lines) == 1, language_lines
    assert "(from " in language_lines[0]
    # Reply/output pins beat primary-language / speaks-lists: German reply
    # replaces the earlier Arabic primary so the standing block agrees with
    # how the assistant should answer.
    assert "German" in language_lines[0]
    assert "Primary language: Arabic" not in card
