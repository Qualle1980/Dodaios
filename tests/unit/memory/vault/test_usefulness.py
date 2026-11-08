"""Being shown is not evidence of having helped, and confidence can fall.

These are failures the AK Matrix memory import plan recorded against its own
store, and the ported vault arrived with them:

- "hit_count currently means surfaced/recalled, not proven useful"
- "mere retrieval must not count as success"
- "Confidence must be able to go down when retrieved memory was irrelevant or
  harmful"
- "Default imported inferences to candidate/unverified"

The always-on profile block used to be ordered by ``use_count``, which recall
increments on every hit. Being in the block is what gets a memory surfaced, so
the block fed itself: whatever it already held climbed, and a better memory
could not get in under the character budget.
"""
from __future__ import annotations

import sqlite3

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault.models import (
    INFERRED_MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    PROFILE_MIN_CONFIDENCE,
    UNHELPFUL_STEP,
    Memory,
    content_hash,
)


@pytest.fixture
def vault(tmp_path):
    with MemoryVault(tmp_path / "vault.db") as handle:
        yield handle


def _add(vault: MemoryVault, text: str, kind: str = "preference") -> Memory:
    result = vault.remember(text, kind)
    assert result.memory is not None
    return result.memory


# -- surfacing is not usefulness ----------------------------------------


def test_being_surfaced_does_not_move_a_memory_up_the_profile_block(vault) -> None:
    first = _add(vault, "I drink my coffee black.")
    _add(vault, "I write my commit messages in English.")
    before = vault.profile_block()

    # Recall surfaced the first memory twenty times and nothing confirmed it.
    for _ in range(20):
        vault.mark_used([first.id])

    assert vault.profile_block() == before
    assert vault.get(first.id).use_count == 20


def test_only_a_confirmed_memory_climbs(vault) -> None:
    low = _add(vault, "I drink my coffee black.")
    high = _add(vault, "I write my commit messages in English.")
    for _ in range(5):
        vault.mark_used([low.id])
    vault.mark_helpful([high.id])

    texts = [line[2:] for line in vault.profile_block().splitlines() if line.startswith("- ")]

    assert texts.index(high.text) < texts.index(low.text)
    stored = vault.get(high.id)
    assert stored.helpful_count == 1
    assert stored.last_helpful_at is not None


def test_the_two_counters_are_independent(vault) -> None:
    memory = _add(vault, "I drink my coffee black.")

    vault.mark_used([memory.id])
    vault.mark_helpful([memory.id])

    stored = vault.get(memory.id)
    assert (stored.use_count, stored.helpful_count) == (1, 1)


def test_marking_nothing_is_harmless(vault) -> None:
    vault.mark_helpful([])
    vault.mark_unhelpful([])

    assert vault.stats()["current"] == 0


# -- confidence can fall ------------------------------------------------


def test_an_unhelpful_mark_lowers_confidence_and_is_audited(vault) -> None:
    memory = _add(vault, "I live in Berlin.", "profile")
    assert memory.confidence == 1.0

    vault.mark_unhelpful([memory.id], reason="the user corrected it")

    stored = vault.get(memory.id)
    assert stored.confidence == pytest.approx(1.0 - UNHELPFUL_STEP)
    assert stored.unhelpful_count == 1
    assert "unhelpful" in [entry["action"] for entry in vault.audit_log()]


def test_enough_marks_push_a_memory_out_of_the_always_on_block(vault) -> None:
    """A signal that changes nothing is theatre: the floor is what makes it real."""
    memory = _add(vault, "I prefer tabs over spaces.")
    assert memory.text in vault.profile_block()

    while vault.get(memory.id).confidence >= PROFILE_MIN_CONFIDENCE:
        vault.mark_unhelpful([memory.id])

    assert memory.text not in vault.profile_block()
    # Still searchable and still in its history: this is not forgetting.
    assert any(hit.memory.id == memory.id for hit in vault.search("tabs spaces"))


def test_the_pin_the_user_set_outranks_the_floor(vault) -> None:
    memory = _add(vault, "I prefer tabs over spaces.")
    vault.pin(memory.id, True)

    for _ in range(10):
        vault.mark_unhelpful([memory.id])

    assert vault.get(memory.id).confidence == pytest.approx(MIN_CONFIDENCE)
    assert memory.text in vault.profile_block(), "a pin is the user's own word"


def test_confidence_stops_above_zero(vault) -> None:
    memory = _add(vault, "I live in Berlin.", "profile")

    for _ in range(50):
        vault.mark_unhelpful([memory.id])

    assert vault.get(memory.id).confidence == pytest.approx(MIN_CONFIDENCE)


# -- an inference enters as a candidate ---------------------------------


@pytest.mark.parametrize("origin", ["inferred", "imported"])
def test_an_inference_cannot_claim_certainty(vault, origin: str) -> None:
    result = vault.remember(
        "The user dislikes meetings.", "preference", origin=origin, confidence=1.0
    )

    assert result.memory is not None
    assert result.memory.confidence == pytest.approx(INFERRED_MAX_CONFIDENCE)


def test_what_the_user_said_keeps_its_certainty(vault) -> None:
    result = vault.remember("I dislike meetings.", "preference", origin="user", confidence=1.0)

    assert result.memory is not None
    assert result.memory.confidence == 1.0


def test_an_inferred_supersede_is_capped_too(vault) -> None:
    stated = vault.remember("I live in Hamburg.", "profile", origin="user")
    assert stated.memory is not None
    assert stated.memory.confidence == 1.0

    changed = vault.supersede(stated.memory.id, "I live in Berlin.", origin="inferred")

    assert changed.memory is not None
    assert changed.memory.confidence == pytest.approx(INFERRED_MAX_CONFIDENCE)


def test_an_imported_row_claiming_certainty_is_capped(vault) -> None:
    text = "The user dislikes meetings."
    smuggled = Memory(
        id="mem_20260101000000_aaaaaaaaaaaa",
        content_hash=content_hash("preference", text),
        kind="preference",
        text=text,
        recorded_at="2026-01-01T00:00:00+00:00",
        origin="imported",
        confidence=1.0,
    )

    assert vault.insert_imported(smuggled) is True
    assert vault.get(smuggled.id).confidence == pytest.approx(INFERRED_MAX_CONFIDENCE)


# -- the schema grows without losing anything ---------------------------


def test_a_file_from_the_previous_version_gains_the_columns(tmp_path) -> None:
    """An ALTER, not a rewrite: rows written before must still be there."""
    path = tmp_path / "vault.db"
    with MemoryVault(path) as vault:
        memory = _add(vault, "I live in Berlin.", "profile")

    # Put the file back into its schema-1 shape.
    db = sqlite3.connect(path)
    for column in ("helpful_count", "last_helpful_at", "unhelpful_count"):
        db.execute(f"ALTER TABLE memory DROP COLUMN {column}")  # noqa: S608
    db.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    db.commit()
    db.close()

    with MemoryVault(path) as reopened:
        stored = reopened.get(memory.id)
        assert stored is not None
        assert stored.text == "I live in Berlin."
        assert (stored.helpful_count, stored.unhelpful_count) == (0, 0)
        assert reopened.get_meta("schema_version") == "2"
        reopened.mark_helpful([memory.id])
        assert reopened.get(memory.id).helpful_count == 1
