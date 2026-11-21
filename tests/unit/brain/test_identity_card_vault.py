"""The personal memory vault rides the standing block, inside its budget.

ADR-0044 wants a small profile block always in the prompt. That block already
exists here: the identity card has the hard character budget, the byte-stable
caching and the "never read this out" contract. So the vault becomes a third
SOURCE of that card rather than a second block - the prompt does not grow by a
character, and a vault memory outranks a line distilled out of a note because
only the memory can be superseded, pinned and forgotten.
"""
from __future__ import annotations

import pytest

from jarvis.brain.identity_card import (
    collect_vault_sections,
    distill_identity_card,
    _hash_sources,
)
from jarvis.memory.vault import MemoryVault, runtime


@pytest.fixture
def vault(tmp_path):
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    try:
        yield handle
    finally:
        runtime.set_active(None)


NOTE = """---
type: entity
slug: ahmad
---

# Ahmad

## Summary

The user.

## Profile

- Languages: German, English
- Role: developer
"""


# -- the card carries memories -----------------------------------------


def test_a_memory_reaches_the_standing_block() -> None:
    card = distill_identity_card(
        vault_sections=[("profile", ["Ahmad lives in Germany."])],
    )

    assert card == "- Ahmad lives in Germany."


def test_memories_come_before_the_notes() -> None:
    """A note line cannot be superseded, pinned or forgotten; a memory can."""
    card = distill_identity_card(
        profile_markdown=NOTE,
        vault_sections=[("pinned", ["Ahmad writes every commit in English."])],
    )

    lines = card.splitlines()
    assert lines[0] == "- Ahmad writes every commit in English."
    assert len(lines) > 1, "the notes still fill the rest of the budget"


def test_the_sections_keep_the_order_the_vault_chose() -> None:
    card = distill_identity_card(
        vault_sections=[
            ("pinned", ["Pinned one."]),
            ("profile", ["Profile one."]),
            ("preference", ["Preference one."]),
        ]
    )

    assert card.splitlines() == ["- Pinned one.", "- Profile one.", "- Preference one."]


def test_an_empty_vault_changes_nothing() -> None:
    with_vault = distill_identity_card(profile_markdown=NOTE, vault_sections=[])
    without = distill_identity_card(profile_markdown=NOTE)

    assert with_vault == without


def test_blank_sentences_are_dropped() -> None:
    card = distill_identity_card(vault_sections=[("profile", ["", "   ", "Real one."])])

    assert card == "- Real one."


def test_whitespace_is_normalised() -> None:
    card = distill_identity_card(vault_sections=[("profile", ["Ahmad  lives\n in Germany."])])

    assert card == "- Ahmad lives in Germany."


# -- and the block does not grow ---------------------------------------


def test_the_budget_is_shared_not_doubled() -> None:
    """The whole point: no extra tokens per turn, and twice over in a live call."""
    budget = 120
    notes_only = distill_identity_card(profile_markdown=NOTE, max_chars=budget)
    both = distill_identity_card(
        profile_markdown=NOTE,
        vault_sections=[("profile", ["Ahmad lives in Germany."])],
        max_chars=budget,
    )

    assert len(notes_only) <= budget
    assert len(both) <= budget
    assert "Ahmad lives in Germany." in both


def test_a_vault_that_fills_the_budget_leaves_the_notes_out() -> None:
    long_memories = [f"Fact number {i} about the user." for i in range(40)]
    card = distill_identity_card(
        profile_markdown=NOTE,
        vault_sections=[("profile", long_memories)],
        max_chars=200,
    )

    assert len(card) <= 200
    assert "Languages" not in card, "the vault outranks the notes under pressure"


def test_one_memory_longer_than_the_budget_yields_an_empty_card() -> None:
    card = distill_identity_card(vault_sections=[("profile", ["x" * 500])], max_chars=50)

    assert card == ""


# -- the cache has to notice a changed memory ---------------------------


def test_the_source_hash_moves_when_a_memory_changes() -> None:
    """Otherwise the card would serve stale text until a note happened to change."""
    before = _hash_sources("", None, [("profile", ["Ahmad lives in Hamburg."])])
    after = _hash_sources("", None, [("profile", ["Ahmad lives in Berlin."])])

    assert before != after


def test_the_source_hash_is_stable_for_the_same_memories() -> None:
    sections = [("profile", ["Ahmad lives in Berlin."])]

    assert _hash_sources("", None, sections) == _hash_sources("", None, sections)


def test_the_hash_separates_the_sentences() -> None:
    """Concatenating without a separator would hash two splits the same."""
    one = _hash_sources("", None, [("profile", ["ab", "c"])])
    other = _hash_sources("", None, [("profile", ["a", "bc"])])

    assert one != other


def test_an_old_call_without_vault_sections_still_hashes() -> None:
    assert _hash_sources("note", None)


# -- reading the live vault --------------------------------------------


def test_the_sections_come_from_the_running_vault(vault) -> None:
    vault.remember("Ahmad lives in Germany.", "profile")
    vault.remember("Everything committed is in English.", "procedural")

    sections = collect_vault_sections(None)

    sentences = [s for _section, group in sections for s in group]
    assert "Ahmad lives in Germany." in sentences
    assert "Everything committed is in English." in sentences


def test_a_sensitive_memory_stays_out_of_the_standing_block(vault) -> None:
    vault.remember("Ahmad lives in Germany.", "profile")
    vault.remember("A private health detail.", "profile", sensitivity="sensitive")

    sentences = [s for _section, group in collect_vault_sections(None) for s in group]

    assert "Ahmad lives in Germany." in sentences
    assert "A private health detail." not in sentences, "it reaches the model via search only"


def test_an_episode_stays_out_of_the_standing_block(vault) -> None:
    vault.remember("Ahmad lives in Germany.", "profile")
    vault.remember("Ahmad went to the dentist on Tuesday.", "episode")

    sentences = [s for _section, group in collect_vault_sections(None) for s in group]

    assert "Ahmad went to the dentist on Tuesday." not in sentences


def test_no_vault_means_no_sections() -> None:
    runtime.set_active(None)

    class Off:
        class memory:  # noqa: N801 - a stand-in for the config object
            class vault:
                enabled = False

    assert collect_vault_sections(Off()) == []


def test_a_broken_vault_costs_the_sections_not_the_turn(monkeypatch) -> None:
    import jarvis.memory.vault.runtime as vault_runtime

    def explode(_config):
        raise RuntimeError("no vault today")

    monkeypatch.setattr(vault_runtime, "active", explode)

    assert collect_vault_sections(None) == []
