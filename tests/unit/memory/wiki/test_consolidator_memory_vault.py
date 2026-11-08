"""A candidate the judge accepted is also filed in the memory vault.

The hook sits where the wiki write has already landed, so the secret guard and
the page validation have passed before a memory is written, and it costs no
extra model call: the vault files the same judged candidate.

It must also be impossible for the vault to undo durable work. A fault while
filing leaves the page written and the journal row closed.
"""
from __future__ import annotations

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.wiki.journal import CandidateFact

from .test_consolidator import (  # noqa: F401 - `stack` is used as a fixture
    LENA_BODY,
    FakeBrain,
    _consolidator,
    _judge_json,
    _write_aged,
    stack,
)


@pytest.fixture
def memory_vault(tmp_path):
    with MemoryVault(tmp_path / "memory-vault.db") as handle:
        yield handle


def _queue(journal, fact: str, *, kind: str = "person", basis: str = "explicit") -> int:
    journal.append(
        [
            CandidateFact(
                fact=fact,
                kind=kind,
                subjects=("lena",),
                evidence_turn_id="turn-1",
                evidence_excerpt="Lena moved to Hamburg last week.",
                basis=basis,
            )
        ],
        source_label="voice-fact:1",
        turn_hash="h1",
    )
    return journal.pending()[0].id


@pytest.mark.asyncio
async def test_an_accepted_add_is_also_a_memory(stack, memory_vault) -> None:
    vault_root, _curator, journal = stack
    cid = _queue(journal, "Lena moved to Hamburg.")
    brain = FakeBrain(
        [
            _judge_json(
                [
                    {
                        "candidate_id": cid,
                        "decision": "add",
                        "target": "entities/lena.md",
                        "new_body": LENA_BODY,
                        "reason": "new person",
                    }
                ]
            )
        ]
    )
    consolidator = _consolidator(stack, brain, vault_provider=lambda: memory_vault)

    await consolidator.run_once()

    assert (vault_root / "entities" / "lena.md").is_file(), "the page write must still happen"
    stored = memory_vault.list()
    assert [m.text for m in stored] == ["Lena moved to Hamburg."]
    memory = stored[0]
    assert memory.kind == "relationship", "a person candidate is a relationship memory"
    assert memory.entities == ("lena",)
    assert memory.origin == "user", "the user asserted it (basis explicit)"
    assert memory.source["turn_id"] == "turn-1"
    assert memory.source["candidate_id"] == cid


@pytest.mark.asyncio
async def test_a_behavioral_candidate_enters_as_an_inference(stack, memory_vault) -> None:
    _vault_root, _curator, journal = stack
    cid = _queue(journal, "Lena visits the animal clinic every week.", basis="behavioral")
    brain = FakeBrain(
        [
            _judge_json(
                [
                    {
                        "candidate_id": cid,
                        "decision": "add",
                        "target": "entities/lena.md",
                        "new_body": LENA_BODY,
                        "reason": "new person",
                    }
                ]
            )
        ]
    )
    consolidator = _consolidator(stack, brain, vault_provider=lambda: memory_vault)

    await consolidator.run_once()

    stored = memory_vault.list()
    assert len(stored) == 1
    assert stored[0].origin == "inferred"
    assert stored[0].confidence < 1.0, "a fact read out of a description is not certain"


@pytest.mark.asyncio
async def test_an_update_writes_the_page_but_closes_no_memory(stack, memory_vault) -> None:
    """An `update` names a page; the judge never saw a memory id, so nothing is closed."""
    vault_root, _curator, journal = stack
    _write_aged(vault_root / "entities" / "lena.md", LENA_BODY)
    standing = memory_vault.remember("Lena lives in Hamburg.", "relationship", origin="user")
    assert standing.memory is not None
    cid = _queue(journal, "Lena got a new job at the animal clinic.")
    brain = FakeBrain(
        [
            _judge_json(
                [
                    {
                        "candidate_id": cid,
                        "decision": "update",
                        "target": "entities/lena.md",
                        "new_body": LENA_BODY + "\n- Works at the animal clinic.\n",
                        "reason": "new job",
                    }
                ]
            )
        ]
    )
    consolidator = _consolidator(stack, brain, vault_provider=lambda: memory_vault)

    await consolidator.run_once()

    still = memory_vault.get(standing.memory.id)
    assert still is not None
    assert still.is_current(memory_vault.now()), "a memory that is still true was closed"
    assert memory_vault.stats()["current"] == 1, "nothing was added either"


@pytest.mark.asyncio
async def test_with_the_vault_off_the_wiki_run_is_unchanged(stack) -> None:
    vault_root, _curator, journal = stack
    cid = _queue(journal, "Lena moved to Hamburg.")
    brain = FakeBrain(
        [
            _judge_json(
                [
                    {
                        "candidate_id": cid,
                        "decision": "add",
                        "target": "entities/lena.md",
                        "new_body": LENA_BODY,
                        "reason": "new person",
                    }
                ]
            )
        ]
    )
    consolidator = _consolidator(stack, brain, vault_provider=lambda: None)

    label = await consolidator.run_once()

    assert label == "journal-batch:1"
    assert (vault_root / "entities" / "lena.md").is_file()
    assert journal.pending() == []


@pytest.mark.asyncio
async def test_a_failing_vault_leaves_the_page_and_the_journal_alone(stack) -> None:
    """The wiki write is durable before the vault is touched; it must stay that way."""
    vault_root, _curator, journal = stack
    cid = _queue(journal, "Lena moved to Hamburg.")

    def explode():
        raise RuntimeError("the vault is on fire")

    brain = FakeBrain(
        [
            _judge_json(
                [
                    {
                        "candidate_id": cid,
                        "decision": "add",
                        "target": "entities/lena.md",
                        "new_body": LENA_BODY,
                        "reason": "new person",
                    }
                ]
            )
        ]
    )
    consolidator = _consolidator(stack, brain, vault_provider=explode)

    label = await consolidator.run_once()

    assert label == "journal-batch:1"
    assert (vault_root / "entities" / "lena.md").is_file()
    assert journal.pending() == [], "the row stays closed; the page is already written"
