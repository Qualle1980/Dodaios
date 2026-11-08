"""A judged candidate becomes a memory, and an ambiguous one becomes nothing.

The wiki's Stage-1 extractor already emits what the vault stores, so filing
costs no model call. What it must never do is guess: ``update`` and
``invalidate`` name a wiki page, and the page-shaped judge never saw a memory
id, so picking the nearest memory would silently close a fact that is still
true.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault.candidates import (
    FALLBACK_KIND,
    KIND_MAP,
    file_candidate,
    source_of,
    vault_kind,
    vault_origin,
)
from jarvis.memory.vault.models import INFERRED_MAX_CONFIDENCE, KINDS


@dataclass
class FakeCandidate:
    """A journal row as the vault reads it (the fields of ``JournalRow``)."""

    id: int = 7
    fact: str = "The user lives in Berlin."
    kind: str = "identity"
    subjects: tuple[str, ...] = ("berlin",)
    evidence_turn_id: str = "turn-42"
    evidence_excerpt: str = "I live in Berlin now."
    session_id: str = "sess-1"
    source_label: str = "voice"
    basis: str = "explicit"


@pytest.fixture
def vault(tmp_path):
    with MemoryVault(tmp_path / "vault.db") as handle:
        yield handle


# -- the two vocabularies line up ---------------------------------------


def test_every_mapped_kind_is_a_real_vault_kind() -> None:
    """A typo here would file memories under a kind the store rejects."""
    assert set(KIND_MAP.values()) <= set(KINDS)
    assert FALLBACK_KIND in KINDS


def test_the_twelve_kinds_stage_one_emits_are_all_mapped() -> None:
    """From the extractor prompt: a new kind there must be added here."""
    emitted = {
        "identity",
        "preference",
        "activity",
        "person",
        "project",
        "decision",
        "event",
        "asset",
        "place",
        "organization",
        "relationship",
        "other",
    }

    assert emitted <= set(KIND_MAP)


def test_an_unknown_kind_is_filed_rather_than_dropped() -> None:
    assert vault_kind("something-new") == FALLBACK_KIND
    assert vault_kind("") == FALLBACK_KIND


def test_evidence_grading_decides_the_origin() -> None:
    assert vault_origin("explicit") == "user"
    assert vault_origin("behavioral") == "inferred"
    # Anything ungraded is treated as a guess, never as the user's own word.
    assert vault_origin("") == "inferred"
    assert vault_origin("something-new") == "inferred"


def test_the_source_can_be_traced_back_to_the_turn() -> None:
    source = source_of(FakeCandidate())

    assert source["candidate_id"] == 7
    assert source["turn_id"] == "turn-42"
    assert source["session_id"] == "sess-1"
    assert source["evidence"] == "I live in Berlin now."


def test_a_long_excerpt_is_cut(vault) -> None:
    source = source_of(FakeCandidate(evidence_excerpt="x" * 5000))

    assert len(source["evidence"]) == 500


# -- add writes ---------------------------------------------------------


def test_an_add_becomes_a_memory_with_its_entities_and_source(vault) -> None:
    result = file_candidate(vault, FakeCandidate(), "add")

    assert result.wrote
    stored = vault.get(result.memory_id)
    assert stored.text == "The user lives in Berlin."
    assert stored.kind == "profile"
    assert stored.entities == ("berlin",)
    assert stored.origin == "user"
    assert stored.source["turn_id"] == "turn-42"


def test_a_behavioral_add_enters_as_a_candidate(vault) -> None:
    """The origin mapping is what makes the confidence cap bite."""
    candidate = FakeCandidate(
        fact="The user plays golf actively with friends.", kind="activity", basis="behavioral"
    )

    result = file_candidate(vault, candidate, "add")

    assert result.wrote
    stored = vault.get(result.memory_id)
    assert stored.origin == "inferred"
    assert stored.confidence == pytest.approx(INFERRED_MAX_CONFIDENCE)


def test_the_same_fact_twice_is_not_duplicated(vault) -> None:
    first = file_candidate(vault, FakeCandidate(), "add")
    second = file_candidate(vault, FakeCandidate(id=8), "add")

    assert first.wrote
    assert second.action == "existing"
    assert vault.stats()["current"] == 1


def test_a_candidate_shaped_like_a_credential_is_refused(vault) -> None:
    secret = "sk-" + "proj-" + "A" * 40
    candidate = FakeCandidate(fact=f"The user's key is {secret}")

    result = file_candidate(vault, candidate, "add")

    assert result.action == "refused"
    assert vault.stats()["current"] == 0


def test_a_candidate_with_no_fact_writes_nothing(vault) -> None:
    result = file_candidate(vault, FakeCandidate(fact="   "), "add")

    assert result.action == "deferred"
    assert vault.stats()["current"] == 0


# -- and the rest does not guess ----------------------------------------


@pytest.mark.parametrize("decision", ["update", "invalidate"])
def test_a_decision_that_names_a_page_closes_nothing(vault, decision: str) -> None:
    """The whole point: no silent supersede of a memory the judge never saw."""
    standing = vault.remember("The user lives in Hamburg.", "profile", origin="user")
    assert standing.memory is not None

    result = file_candidate(vault, FakeCandidate(), decision)

    assert result.action == "deferred"
    assert "no memory ids" in result.reason
    still = vault.get(standing.memory.id)
    assert still.is_current(vault.now()), "a memory that is still true was closed"
    assert vault.stats()["current"] == 1


def test_a_noop_writes_nothing(vault) -> None:
    result = file_candidate(vault, FakeCandidate(), "noop")

    assert result.action == "deferred"
    assert vault.stats()["current"] == 0


def test_an_unknown_decision_writes_nothing(vault) -> None:
    result = file_candidate(vault, FakeCandidate(), "something-else")

    assert result.action == "deferred"
    assert vault.stats()["current"] == 0


def test_a_forgotten_fact_is_not_brought_back_by_an_inference(vault) -> None:
    """`forget` is the user's word; a later inferred candidate must not undo it."""
    stored = file_candidate(vault, FakeCandidate(basis="behavioral"), "add")
    assert stored.wrote
    assert vault.forget(stored.memory_id) == 1

    again = file_candidate(vault, FakeCandidate(id=9, basis="behavioral"), "add")

    assert again.action == "blocked"
    assert vault.stats()["current"] == 0
