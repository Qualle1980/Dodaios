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


def test_a_line_that_merely_names_a_language_is_not_the_language_line() -> None:
    """Mentioning a language is not stating which one the user speaks.

    The first version of this collector matched a bare language name anywhere,
    so a commit rule took the language line's place: it got the provenance
    suffix and the real language line behind it was dropped as a duplicate --
    the contradictory-key bug this code exists to prevent. Caught by
    test_identity_card_vault::test_memories_come_before_the_notes in CI.
    """
    card = ic.distill_identity_card(
        vault_sections=[
            (
                "pinned",
                [
                    "Ahmad writes every commit in English.",
                    "Reply language: German",
                ],
            )
        ],
    )

    commit_line = next(line for line in card.splitlines() if "commit" in line)
    assert commit_line == "- Ahmad writes every commit in English.", "no provenance suffix"
    assert "- Reply language: German (from memory-vault)" in card, "the real one survives"


def test_a_german_language_line_is_recognised_in_german() -> None:
    card = ic.distill_identity_card(
        vault_sections=[("pinned", ["Hauptsprache: Deutsch", "Primary language: Arabic"])],
    )

    kept = [  # i18n-allow: German cue asserted in the card text
        line
        for line in card.splitlines()
        if "sprache" in line.casefold() or "Arabic" in line  # i18n-allow: German cue
    ]
    assert len(kept) == 1, kept
    assert "(from memory-vault)" in kept[0]
