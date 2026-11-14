"""Per-kind caps on the standing profile block (brain slice 1).

Without caps, profile + pinned can spend the whole character budget before
procedural / project / preference appear. A hand-set procedural fact then
falls out of the always-on block while imported profile fillers remain.
"""

from __future__ import annotations

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault.turn_recall import turn_block

HANDSET_TEXT = (
    "Handset truth: commit messages stay in English and code comments stay English."
)
PINNED_TEXT = "Pinned project: Dodaios memory vault is the personal fact store."
DROPPED_REFERENCE_TEXT = (
    "Imported candidate 00005: synthetic note about topic-5 in category reference."
)

# Acceptance caps from research §R3.6 / order §3.2. Pinned has no line cap.
EXPECTED_KIND_CAPS = {
    "profile": 6,
    "preference": 5,
    "procedural": 5,
    "relationship": 3,
    "project": 3,
    "reference": 0,
}

_KIND_CYCLE = (
    "profile",
    "preference",
    "procedural",
    "relationship",
    "project",
    "reference",
    "reference",
    "episode",
)


def _cap_sum() -> int:
    return sum(EXPECTED_KIND_CAPS.values())


def _build_scale_vault(vault: MemoryVault, n: int = 500) -> None:
    """Synthetic mix matching `_bench/memscale_bench.py` (handset + pin + imports)."""
    vault.remember(
        HANDSET_TEXT,
        kind="procedural",
        origin="user",
        confidence=1.0,
        entities=("Dodaios",),
    )
    vault.remember(
        PINNED_TEXT,
        kind="project",
        origin="user",
        confidence=1.0,
        pinned=True,
        entities=("Dodaios",),
    )
    for i in range(2, n):
        kind = _KIND_CYCLE[i % len(_KIND_CYCLE)]
        text = (
            f"Imported candidate {i:05d}: synthetic note about topic-{i % 97} "
            f"in category {kind}."
        )
        vault.remember(
            text,
            kind=kind,
            origin="imported",
            confidence=0.7,
            entities=(f"Topic{i % 97}",),
        )


def _selection_texts(vault: MemoryVault, max_chars: int = 4000) -> list[str]:
    return [
        memory.text
        for _section, memories in vault.profile_selection(max_chars=max_chars)
        for memory in memories
    ]


def _selection_by_kind(vault: MemoryVault, max_chars: int = 4000) -> dict[str, int]:
    return {
        section: len(memories)
        for section, memories in vault.profile_selection(max_chars=max_chars)
    }


@pytest.fixture
def vault(tmp_path) -> MemoryVault:
    handle = MemoryVault(tmp_path / "vault.db")
    yield handle
    handle.close()


@pytest.fixture
def scale_vault(vault: MemoryVault) -> MemoryVault:
    _build_scale_vault(vault, n=500)
    return vault


# -- §4 acceptance 1–6 (N=500 scale fixture) ---------------------------------


def test_scale_fixture_keeps_pinned_in_standing_block(scale_vault: MemoryVault) -> None:
    """§4.2 — the pinned project line survives the standing budget."""
    assert PINNED_TEXT in _selection_texts(scale_vault, max_chars=4000)


def test_scale_fixture_keeps_handset_procedural_after_caps(
    scale_vault: MemoryVault,
) -> None:
    """§4.3 — hand-set procedural must not be crowded out by imported profile.

    This is the core defect: at N=500 without per-kind caps the handset is
    absent from the 4000-char block. Caps must restore it.
    """
    assert HANDSET_TEXT in _selection_texts(scale_vault, max_chars=4000)


def test_scale_fixture_includes_zero_unpinned_reference_lines(
    scale_vault: MemoryVault,
) -> None:
    """§4.4 — reference is lookup material; standing selection keeps none."""
    by_kind = _selection_by_kind(scale_vault, max_chars=4000)
    assert by_kind.get("reference", 0) == 0


def test_scale_fixture_line_count_respects_kind_caps_plus_pins(
    scale_vault: MemoryVault,
) -> None:
    """§4.5 — fitted lines ≤ sum of kind caps + pinned (pinned uncapped)."""
    sections = scale_vault.profile_selection(max_chars=4000)
    pinned_count = 0
    kind_lines = 0
    for section, memories in sections:
        if section == "pinned":
            pinned_count += len(memories)
        else:
            kind_lines += len(memories)
            assert len(memories) <= EXPECTED_KIND_CAPS[section]
    assert kind_lines + pinned_count == sum(len(m) for _, m in sections)
    assert kind_lines <= _cap_sum()


def test_scale_fixture_dropped_reference_still_searchable(
    scale_vault: MemoryVault,
) -> None:
    """§4.6 — eviction from the block must not erase the row from search."""
    texts = _selection_texts(scale_vault, max_chars=4000)
    assert DROPPED_REFERENCE_TEXT not in texts
    hits = scale_vault.search("Imported candidate 00005")
    assert any(hit.memory.text == DROPPED_REFERENCE_TEXT for hit in hits)


# -- §4.7 pinned reference outranks the zero cap ----------------------------


def test_pinned_reference_appears_despite_zero_kind_cap(vault: MemoryVault) -> None:
    """§4.7 — a pinned reference is the user's word and must still appear."""
    vault.remember(
        "Pinned where: keys hang by the door.",
        kind="reference",
        pinned=True,
        origin="user",
        confidence=1.0,
    )
    vault.remember(
        "Unpinned where: spare key under the mat.",
        kind="reference",
        origin="imported",
        confidence=0.7,
    )
    # Flood early kinds so a char-budget-only path would still have room for
    # references only if they were uncapped; the zero cap must drop unpinned.
    for i in range(8):
        vault.remember(
            f"Profile filler {i:02d} about the synthetic user identity line.",
            kind="profile",
            origin="imported",
            confidence=0.7,
        )

    texts = _selection_texts(vault, max_chars=4000)
    by_kind = _selection_by_kind(vault, max_chars=4000)

    assert "Pinned where: keys hang by the door." in texts
    assert "Unpinned where: spare key under the mat." not in texts
    assert by_kind.get("reference", 0) == 0


# -- §4.8 standing_ids shrink so turn recall can surface the rest ------------


def test_capped_out_line_is_freed_for_turn_block(vault: MemoryVault) -> None:
    """§4.8 — a line off the standing block may appear in per-turn recall.

    `runtime.turn_suffix` builds `standing_ids` from `profile_selection()`.
    Smaller caps shrink that set, so a dropped fact becomes reachable again.
    """
    stored = vault.remember(
        "The vault backup lives on DriveX unique-token-xyz.",
        kind="reference",
        origin="user",
        confidence=1.0,
        entities=("DriveX",),
    ).memory
    assert stored is not None

    standing = frozenset(
        memory.id
        for _section, memories in vault.profile_selection(max_chars=4000)
        for memory in memories
    )
    assert stored.id not in standing, "uncapped reference must leave standing_ids"

    block, shown = turn_block(
        vault,
        "Where is the DriveX unique-token-xyz backup?",
        standing_ids=standing,
    )
    assert "DriveX unique-token-xyz" in block
    assert stored.id in shown


def test_kind_cap_truncation_is_logged(vault: MemoryVault, caplog: pytest.LogCaptureFixture) -> None:
    """A kind cut by its line cap must not fail silently (AGENTS silence rule)."""
    for i in range(10):
        vault.remember(
            f"Profile identity detail number {i:02d} for the standing block.",
            kind="profile",
            origin="imported",
            confidence=0.7,
        )

    with caplog.at_level("INFO", logger="jarvis.memory.vault.store"):
        vault.profile_selection(max_chars=4000)

    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "capped" in joined.lower()
    assert "profile" in joined.lower()
