"""Memories that match the message, and nothing the prompt already carries.

The standing block answers "who is this person?" under a budget that always
fits. This answers "did they tell me something about what they just asked?",
which no fixed budget can hold. Two rules earn it its tokens: never repeat the
standing block, and only show hits close to the best one - keyword search
always returns something.
"""
from __future__ import annotations

import pytest

from jarvis.memory.vault import MemoryVault, runtime
from jarvis.memory.vault.turn_recall import (
    HEADER,
    TURN_LIMIT,
    content_words,
    turn_block,
)


@pytest.fixture
def vault(tmp_path):
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    try:
        yield handle
    finally:
        runtime.set_active(None)


# -- what counts as a topic --------------------------------------------


def test_filler_words_are_not_topics() -> None:
    assert content_words("What do you think about that?") == ["think"]


def test_german_filler_is_not_a_topic() -> None:
    """Folding turns the German sharp s into a double s, so the list is folded too."""
    assert content_words("Was weißt du über meine Wohnung in Berlin?") == [  # i18n-allow
        "wohnung",
        "berlin",
    ]


def test_short_words_are_dropped() -> None:
    assert content_words("Is it ok?") == []


def test_words_keep_their_order_without_repeats() -> None:
    assert content_words("Berlin, Berlin and Hamburg") == ["berlin", "hamburg"]


def test_a_message_of_pure_filler_yields_nothing() -> None:
    assert content_words("und das ist nicht das was ich meine") == []  # i18n-allow


# -- the block ---------------------------------------------------------


def test_a_matching_memory_is_offered(vault) -> None:
    vault.remember("Ahmad lives in Berlin.", "profile", entities=["berlin"])

    block, shown = turn_block(vault, "How is the weather in Berlin?")

    assert HEADER in block
    assert "Ahmad lives in Berlin." in block
    assert len(shown) == 1


def test_a_message_about_nothing_stored_offers_nothing(vault) -> None:
    vault.remember("Ahmad lives in Berlin.", "profile")

    block, shown = turn_block(vault, "How do I compile a kernel module?")

    assert block == ""
    assert shown == []


def test_a_message_without_content_words_does_not_even_search(vault) -> None:
    class Counting:
        calls = 0

        def search(self, *_args, **_kwargs):
            self.calls += 1
            return []

    counting = Counting()

    assert turn_block(counting, "Is it ok?") == ("", [])
    assert counting.calls == 0, "a message with no topic must not hit the index"


def test_the_standing_block_is_not_repeated(vault) -> None:
    """A memory in the always-on card would otherwise be paid for twice."""
    stored = vault.remember("Ahmad lives in Berlin.", "profile", entities=["berlin"])
    assert stored.memory is not None

    block, shown = turn_block(
        vault, "Something about Berlin", standing_ids=frozenset({stored.memory.id})
    )

    assert block == ""
    assert shown == []


def test_a_weak_hit_next_to_a_strong_one_is_left_out(vault) -> None:
    """Keyword search always answers; only what is close to the best is worth a line."""
    for i in range(12):
        vault.remember(f"Ahmad owns gadget number {i}.", "reference", entities=["gadget"])
    vault.remember("Ahmad lives in Berlin.", "profile", entities=["berlin"])

    block, shown = turn_block(vault, "Berlin")

    assert "Ahmad lives in Berlin." in block
    assert len(shown) <= TURN_LIMIT


def test_the_block_respects_its_character_budget(vault) -> None:
    for i in range(20):
        vault.remember(f"Ahmad knows fact {i} about Berlin and its districts.", "reference",
                       entities=["berlin"])

    block, shown = turn_block(vault, "Berlin districts", max_chars=200)

    body = block[len(HEADER) :]
    assert len(body) <= 201
    assert shown


def test_at_most_the_limit_is_shown(vault) -> None:
    for i in range(20):
        vault.remember(f"Ahmad knows Berlin fact {i}.", "reference", entities=["berlin"])

    _block, shown = turn_block(vault, "Berlin", limit=3)

    assert len(shown) == 3


def test_an_import_does_not_claim_a_start_date(vault) -> None:
    """Its `valid_from` is when it was imported, which would read as precision."""
    vault.remember("Ahmad lives in Berlin.", "profile", origin="imported", entities=["berlin"])

    block, _shown = turn_block(vault, "Berlin")

    assert "since" not in block


def test_a_stated_memory_says_since_when(vault) -> None:
    vault.remember("Ahmad lives in Berlin.", "profile", origin="user", entities=["berlin"])

    block, _shown = turn_block(vault, "Berlin")

    assert "(since " in block


def test_a_search_fault_costs_the_block_not_the_turn(vault) -> None:
    class Broken:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("index on fire")

    assert turn_block(Broken(), "Berlin") == ("", [])


# -- through the runtime, which is what the prompt calls ----------------


def test_the_runtime_counts_a_shown_memory_as_surfaced(vault) -> None:
    """An episode is never in the standing block, so only the turn can offer it."""
    stored = vault.remember(
        "Ahmad met the notary in Berlin on 2026-10-02.", "episode", entities=["berlin"]
    )
    assert stored.memory is not None

    suffix = runtime.turn_suffix(None, "What happened in Berlin?")

    assert "notary" in suffix
    after = vault.get(stored.memory.id)
    assert after.use_count == 1
    assert after.helpful_count == 0, "shown is not helped (principle 7)"


def test_the_runtime_leaves_out_what_the_standing_block_holds(vault) -> None:
    """End to end: one memory, in the card, must not appear twice."""
    vault.remember("Ahmad lives in Berlin.", "profile", entities=["berlin"])

    suffix = runtime.turn_suffix(None, "Tell me about Berlin")

    # A single profile memory is in the standing selection, so the turn block
    # has nothing left to add.
    assert suffix == ""


def test_the_runtime_says_nothing_on_an_empty_message(vault) -> None:
    vault.remember("Ahmad lives in Berlin.", "profile")

    assert runtime.turn_suffix(None, "   ") == ""


def test_the_runtime_says_nothing_with_the_vault_off(vault) -> None:
    class Off:
        class memory:  # noqa: N801 - a stand-in for the config object
            class vault:
                enabled = False

    runtime.set_active(None)
    assert runtime.turn_suffix(Off(), "Berlin") == ""
