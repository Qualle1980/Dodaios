"""The standing block reaches a call, not only the chat.

ADR-0044 asks for the same block on every surface. It was on one: the typed
brain has carried the identity card for a while, while a live session and a
realtime session each build their own instruction stack and had no card at
all. So a call could not use a standing fact about the person it was speaking
to, and the person heard it ask for something it already knew.

Same class of gap as the vault search tool before #123, and the same cause:
three prompt builds, one of them wired.

Voice gets the COMPACT budget. A live or realtime session re-pays its
instructions on every turn because the provider holds the session context, so
half the typed budget is the honest price.
"""
from __future__ import annotations

import pytest

from jarvis.brain.identity_card import (
    MAX_IDENTITY_CARD_CHARS,
    MIN_COMPACT_IDENTITY_CARD_CHARS,
    _budget,
    identity_card_block,
    reset_identity_card_cache,
)
from jarvis.live.session import _identity as live_identity
from jarvis.live.session import _standing_memory
from jarvis.memory.vault import MemoryVault, runtime
from jarvis.realtime.session import _identity_block as realtime_identity
from jarvis.realtime.session import _standing_memory_block


@pytest.fixture
def vault(tmp_path):
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    reset_identity_card_cache()
    try:
        yield handle
    finally:
        runtime.set_active(None)
        reset_identity_card_cache()


@pytest.fixture
def remembered(vault):
    vault.remember("Ahmad lives in Germany.", "profile")
    vault.remember("Everything committed is written in English.", "procedural")
    return vault


# -- every surface carries it ------------------------------------------


def test_the_typed_brain_carries_the_block(remembered) -> None:
    block = identity_card_block(None)

    assert "Ahmad lives in Germany." in block


def test_a_live_call_carries_the_block(remembered) -> None:
    """Through `_identity`, the function the session actually sends.

    Asserting on `_standing_memory` alone passes with the call unwired - which
    it did, the first time this test was written.
    """
    instructions = live_identity(None)

    assert "Ahmad lives in Germany." in instructions


def test_a_realtime_session_carries_the_block(remembered) -> None:
    """Through `_identity_block`, the function the session actually sends."""
    instructions = realtime_identity(None)

    assert "Ahmad lives in Germany." in instructions


def test_all_three_say_the_same_thing(remembered) -> None:
    """One source, so a fact cannot be true in the chat and missing in a call."""
    typed = identity_card_block(None, compact=True).strip()

    assert typed
    assert typed in live_identity(None)
    assert typed in realtime_identity(None)


def test_the_live_block_is_appended_not_prepended(remembered) -> None:
    """It joins the instruction stack; it does not displace the character."""
    assert _standing_memory(None).startswith("\n\n")


# -- and voice pays less -----------------------------------------------


def test_voice_gets_half_the_typed_budget() -> None:
    assert _budget(None, compact=False) == MAX_IDENTITY_CARD_CHARS
    assert _budget(None, compact=True) == MAX_IDENTITY_CARD_CHARS // 2


def test_the_compact_budget_has_a_floor() -> None:
    """Half of a small budget must not drop the person's own name."""

    class Tight:
        class wiki_context:  # noqa: N801 - a stand-in for the config object
            identity_card_max_chars = 600

    assert _budget(Tight(), compact=True) == MIN_COMPACT_IDENTITY_CARD_CHARS


def test_a_configured_budget_below_the_floor_is_not_raised_by_it() -> None:
    """The floor must never hand voice MORE than the person configured."""

    class Tiny:
        class wiki_context:  # noqa: N801
            identity_card_max_chars = 120

    assert _budget(Tiny(), compact=True) == 120


def test_the_two_budgets_do_not_evict_each_other(remembered) -> None:
    full = identity_card_block(None, compact=False)
    compact = identity_card_block(None, compact=True)

    assert identity_card_block(None, compact=False) == full
    assert identity_card_block(None, compact=True) == compact


# -- and nothing breaks a call -----------------------------------------


def test_the_block_is_switched_off_with_the_identity_card(remembered) -> None:
    class Off:
        class wiki_context:  # noqa: N801
            identity_card = False

    assert identity_card_block(Off()) == ""


def test_an_empty_vault_and_no_notes_add_nothing(vault) -> None:
    assert _standing_memory(None) == ""
    assert _standing_memory_block(None) == ""


def test_a_fault_costs_the_block_not_the_call(monkeypatch) -> None:
    import jarvis.brain.identity_card as card

    def explode(*_args, **_kwargs):
        raise RuntimeError("no card today")

    monkeypatch.setattr(card, "identity_card_block", explode)

    assert _standing_memory(None) == ""
    assert _standing_memory_block(None) == ""
