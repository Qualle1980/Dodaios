"""A spoken "I cannot do that" is measured against the live tool surface.

The live path cannot take a sentence back: the model speaks audio directly,
so by the time the denial is in the transcript the user has already heard
it. These tests pin the DETECTOR only — how often a denial contradicts the
attached tools — which is what decides whether the intrusive remedies are
worth building.
"""

from __future__ import annotations

import pytest

from jarvis.brain.action_honesty import capability_denial_terms
from jarvis.brain.evidence_gate import live_surface_covers


def _contradicted(reply: str, tool_names: list[str]) -> bool:
    terms = capability_denial_terms(reply)
    return bool(terms) and live_surface_covers(terms, tool_names)


# The live defect, verbatim from the 2026-11-07 voice session: search_web
# was callable and 46 GitHub tools were connected.
LIVE_DENIAL = (
    "Ich kann nicht direkt im Internet oder in externen Repos suchen."  # i18n-allow: DE output
)


def test_the_live_denial_is_detected_and_contradicted() -> None:
    assert _contradicted(LIVE_DENIAL, ["search_web", "github_search_repositories"])


def test_the_same_denial_stands_when_nothing_covers_it() -> None:
    """Without a matching tool the sentence is honest and must not be flagged."""
    assert not _contradicted(LIVE_DENIAL, ["take_appshot", "end_call"])


@pytest.mark.parametrize(
    "reply",
    [
        "Ich habe keinen Zugriff auf deinen Kalender.",  # i18n-allow: DE output
        "Leider nein -- an den Kalender komme ich nicht.",  # i18n-allow: DE output
        "I don't have access to your calendar.",
        "I can't look at your calendar.",
    ],
)
def test_denial_vocabulary(reply: str) -> None:
    assert _contradicted(reply, ["google_calendar"])


@pytest.mark.parametrize(
    "reply",
    [
        # A refusal about the WORLD, not about Jarvis's reach.
        "Das Konzert ist ausverkauft, da kommst du nicht mehr rein.",  # i18n-allow: DE output
        # A plain answer that names the system without denying anything.
        "In deinem Kalender steht morgen um neun ein Termin.",  # i18n-allow: DE output
        "I searched the web and found three results.",
        "",
    ],
)
def test_not_a_capability_denial(reply: str) -> None:
    assert capability_denial_terms(reply) == ()


def test_a_conditional_denial_is_not_a_capability_claim() -> None:
    """"I can't do that WITHOUT X" names a precondition, not a missing tool."""
    assert capability_denial_terms(
        "Ohne deine Freigabe kann ich den Kalender nicht aendern."  # i18n-allow: DE output
    ) == ()
