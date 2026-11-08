"""The three read tools answer from the vault, and never hand out an id.

`memory-topics` is the one upstream has no equivalent for: search only helps
someone who already knows what to search for, so "what do you know about me?"
had no cheap answer at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from jarvis.memory.vault import MemoryVault, runtime
from jarvis.plugins.tool.memory_recall import (
    MemoryHistoryTool,
    MemorySearchTool,
    MemoryTopicsTool,
)


@dataclass
class FakeContext:
    """These tools read nothing from the context; it is passed through."""

    user_utterance: str = ""


@pytest.fixture
def vault(tmp_path):
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    try:
        yield handle
    finally:
        runtime.set_active(None)


@pytest.fixture
def filled(vault):
    vault.remember("Ahmad lives in Germany.", "profile", entities=["ahmad", "germany"])
    vault.remember(
        "Everything committed is written in English.", "procedural", entities=["ahmad"]
    )
    vault.remember("Dodaios is the voice assistant project.", "project", entities=["dodaios"])
    return vault


async def _run(tool, **args) -> str:
    result = await tool.execute(args, FakeContext())
    assert result.success, result.error
    return str(result.output)


# -- search -------------------------------------------------------------


async def test_search_finds_a_fact_and_says_since_when(filled) -> None:
    output = await _run(MemorySearchTool(), query="commit English")

    assert "Everything committed is written in English." in output
    assert "since " in output


async def test_search_says_so_when_it_knows_nothing(filled) -> None:
    output = await _run(MemorySearchTool(), query="submarine racing")

    assert "Nothing remembered" in output


async def test_search_needs_a_query(filled) -> None:
    result = await MemorySearchTool().execute({}, FakeContext())

    assert result.success is False
    assert "query" in (result.error or "")


async def test_search_counts_the_hit_as_surfaced_not_as_helpful(filled) -> None:
    """Principle 7: being shown must not promote a memory into the profile block."""
    before = filled.profile_block()

    await _run(MemorySearchTool(), query="commit English")

    hit = next(m for m in filled.list() if "English" in m.text)
    assert hit.use_count == 1
    assert hit.helpful_count == 0
    assert filled.profile_block() == before


async def test_an_inference_is_marked_as_not_stated_outright(vault) -> None:
    vault.remember("The user plays golf.", "profile", origin="inferred", entities=["golf"])

    output = await _run(MemorySearchTool(), query="golf")

    assert "not stated outright" in output


# -- history ------------------------------------------------------------


async def test_history_answers_where_the_user_lived_before(vault) -> None:
    first = vault.remember("Ahmad lives in Hamburg.", "profile", entities=["ahmad"])
    assert first.memory is not None
    vault.supersede(first.memory.id, "Ahmad lives in Berlin.", origin="user")

    output = await _run(MemoryHistoryTool(), topic="ahmad")

    assert "Hamburg" in output and "Berlin" in output
    assert "[no longer true]" in output
    # Oldest first, so the timeline reads forwards.
    assert output.index("Hamburg") < output.index("Berlin")


async def test_history_falls_back_to_a_search_for_an_unknown_topic(filled) -> None:
    """A topic that is not a recorded entity still gets an answer, just unordered."""
    output = await _run(MemoryHistoryTool(), topic="English")

    assert "No timeline" in output
    assert "Everything committed is written in English." in output


async def test_history_says_so_when_it_knows_nothing(filled) -> None:
    output = await _run(MemoryHistoryTool(), topic="submarine racing")

    assert "Nothing remembered" in output


async def test_history_needs_a_topic(filled) -> None:
    result = await MemoryHistoryTool().execute({}, FakeContext())

    assert result.success is False


# -- topics -------------------------------------------------------------


async def test_topics_lists_the_kinds_and_the_busiest_subjects(filled) -> None:
    output = await _run(MemoryTopicsTool())

    assert "3 memories" in output
    assert "profile 1" in output and "procedural 1" in output and "project 1" in output
    # ahmad appears in two memories, so it leads.
    assert "ahmad (2)" in output
    assert output.index("ahmad") < output.index("dodaios")
    assert "memory-search" in output, "it must say how to go deeper"


async def test_topics_says_so_while_the_vault_is_empty(vault) -> None:
    output = await _run(MemoryTopicsTool())

    assert "still empty" in output


async def test_topics_takes_no_arguments(filled) -> None:
    assert MemoryTopicsTool().input_schema["properties"] == {}


# -- what the model must never see --------------------------------------


@pytest.mark.parametrize(
    "tool, args",
    [
        (MemorySearchTool(), {"query": "ahmad Germany"}),
        (MemoryHistoryTool(), {"topic": "ahmad"}),
        (MemoryTopicsTool(), {}),
    ],
)
async def test_no_tool_hands_the_model_a_memory_id(filled, tool, args) -> None:
    """An id is opaque and stable, and a model shown one reads it out in a call."""
    output = await _run(tool, **args)

    assert "mem_" not in output
    assert all(m.id not in output for m in filled.list())


# -- with the vault off ------------------------------------------------


@pytest.mark.parametrize(
    "tool, args",
    [
        (MemorySearchTool(), {"query": "anything"}),
        (MemoryHistoryTool(), {"topic": "anything"}),
        (MemoryTopicsTool(), {}),
    ],
)
async def test_every_tool_answers_plainly_when_memory_is_off(monkeypatch, tool, args) -> None:
    """Switched off is an answer, not an error the model has to interpret."""
    import jarvis.plugins.tool.memory_recall as module

    monkeypatch.setattr(module, "_vault", lambda: None)

    output = await _run(tool, **args)

    assert "switched off" in output
