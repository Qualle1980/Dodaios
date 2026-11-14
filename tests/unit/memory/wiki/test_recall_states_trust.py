"""Recall must say how much weight a hit deserves.

The vault is searched in full on purpose: a lesson exists so a mistake is not
repeated, and the verified-only gate hid 209 of 216 notes, which defeats that.
Honesty is carried per hit instead — if the model is handed an unconfirmed
note without being told, it states it as fact.
"""
from __future__ import annotations

from pathlib import Path

from jarvis.plugins.tool.wiki_recall import _trust_label


class _Hit:
    def __init__(self, frontmatter: str) -> None:
        self.frontmatter = frontmatter
        self.title = "a note"
        self.snippet = ""
        self.path = Path("a-note.md")


def test_a_verified_note_is_named_verified() -> None:
    assert _trust_label(_Hit("project dodaios __fm_status_verified__")) == "verified"


def test_a_proposed_note_is_named_unconfirmed() -> None:
    assert _trust_label(_Hit("__fm_status_proposed__ project dodaios")) == "unconfirmed"


def test_a_superseded_note_says_so() -> None:
    assert _trust_label(_Hit("__fm_status_superseded__")) == "superseded"


def test_a_note_without_a_status_is_never_called_verified() -> None:
    label = _trust_label(_Hit("project dodaios component live"))
    assert label == "status unknown"
    assert "verified" not in label


def test_the_word_verified_in_another_field_does_not_pass_as_a_status() -> None:
    # The marker exists precisely so prose cannot impersonate a status.
    assert _trust_label(_Hit("summary the fix was verified by ear")) == "status unknown"


def test_a_hit_without_frontmatter_at_all_is_handled() -> None:
    class _Bare:
        title = "bare"
        snippet = ""
        path = Path("bare.md")

    assert _trust_label(_Bare()) == "status unknown"
