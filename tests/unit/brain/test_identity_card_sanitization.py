"""Identity card sanitization: dedupe, language conflict, junk names, third parties.

Invented fixture shapes mirror the live 2026-11-14 card defects without any
real user data. The standing block is paid on every live turn, so the card
must get shorter and safer, not longer.
"""

from __future__ import annotations

from jarvis.brain import identity_card as ic

# Invented, harmless fixture — same ERROR FORMS as the live card, never real data.
_MESSY_PROFILE = """---
type: entity
entity_kind: person
slug: river-user
---

# River Example

## Identity

- Name: What is my nickname again?
- Name: User
- Name: River Example
- Birthday: 12 March 1990
- Date of birth: 12 March 1990
- Birth date: March 12, 1990
- Primary language: Arabic
- Speaks Arabic, German and English

## Preferences

- Reply language: German
- Favourite dish: lentil soup
- Favorite dish: lentil soup
- Short answers

## Relationships

- Partner Alex prefers quiet evenings and has a chronic condition
- Friend Sam is exploring their orientation and lives nearby

## Active projects

- Project: OtherApp 2 — build widgets with DK
- Project: Dodaios — personal voice assistant
"""


def test_messy_fixture_is_deduped_and_sanitized() -> None:
    before = ic.distill_identity_card(profile_markdown=_MESSY_PROFILE)
    # Sanity: without filters the Relationships section used to land here;
    # the fixture is large enough that a naive card would keep third parties.
    assert "River Example" in before

    card = before
    after_chars = len(card)

    # Real name kept; junk names dropped.
    assert "River Example" in card
    assert "What is my nickname again?" not in card
    assert "Name: User" not in card
    assert "- Name: User" not in card

    # Field duplicates collapsed to one.
    assert card.lower().count("birthday") + card.lower().count("birth date") + card.lower().count(
        "date of birth"
    ) <= 1
    assert card.lower().count("lentil soup") == 1

    # One language decision with provenance — not Arabic+German+three languages.
    language_lines = [
        line
        for line in card.splitlines()
        if "language" in line.casefold()
        or "speaks" in line.casefold()
        or "arabic" in line.casefold()
        or "german" in line.casefold()
    ]
    assert len(language_lines) == 1, language_lines
    assert "(from " in language_lines[0]

    # Third-party personal facts stay out of the standing block.
    assert "Partner Alex" not in card
    assert "Friend Sam" not in card
    assert "chronic" not in card.casefold()
    assert "orientation" not in card.casefold()

    # Card stays bounded; record length for the report.
    assert after_chars <= ic.MAX_IDENTITY_CARD_CHARS
    print(f"identity-card sanitization chars after={after_chars}")


def test_sanitized_card_is_shorter_than_unfiltered_concatenation() -> None:
    """Compare against a naive join of the same fixture lines (pre-filter size)."""
    raw_lines = []
    for line in _MESSY_PROFILE.splitlines():
        cleaned = ic._clean(line)
        if cleaned and not cleaned.startswith("#") and not cleaned.startswith("---"):
            raw_lines.append(f"- {cleaned}")
    naive = "\n".join(raw_lines)
    card = ic.distill_identity_card(profile_markdown=_MESSY_PROFILE)
    assert len(card) < len(naive), (
        f"sanitized card must shrink vs naive join: "
        f"card={len(card)} naive={len(naive)}"
    )
    print(f"identity-card chars before_naive={len(naive)} after={len(card)}")


def test_relationships_section_is_not_in_priority() -> None:
    assert "Relationships" not in ic.PROFILE_SECTION_PRIORITY


def test_vault_third_party_sentences_are_dropped() -> None:
    card = ic.distill_identity_card(
        vault_sections=[
            (
                "people",
                [
                    "User River builds Dodaios",
                    "Partner Alex has a chronic condition",
                ],
            )
        ]
    )
    assert "builds Dodaios" in card
    assert "Partner Alex" not in card
    assert "chronic" not in card.casefold()
