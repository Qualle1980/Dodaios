"""The standing block must be a function of the vault, not of a coin flip.

`profile_selection` ranks by (helpful_count, confidence, recorded_at). In
production all three tie constantly: `mark_helpful` has no production caller,
so `helpful_count` is 0 everywhere, imported facts share one confidence, and
`recorded_at` has one-second resolution. Python's sort is stable, so a tie kept
whatever order `list()` returned -- and that query ends `ORDER BY
m.recorded_at DESC, m.id DESC`, where an id carries `secrets.token_hex(6)`.

With the per-kind line cap from #164 that random id decides WHICH facts reach
the model. Measured 2026-11-15: the same unchanged test passed, passed, then
failed on three consecutive runs
(`test_legacy_import.py::test_a_sensitive_fact_stays_out_of_the_standing_block`),
and it blocked every pull request in the repo.

Third time at this exact spot: #130 fixed the same one-second-plus-random-hex
ordering in `about()` and `history()`. `profile_selection` was not part of it.
"""
from __future__ import annotations

import itertools

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault import store as store_module

#: More profile facts than `_PROFILE_KIND_CAPS["profile"]`, so the cap has to
#: drop some and the choice becomes visible.
_FACTS = (
    "Alpha fact.",
    "Bravo fact.",
    "Charlie fact.",
    "Delta fact.",
    "Echo fact.",
    "Foxtrot fact.",
    "Golf fact.",
    "Hotel fact.",
)


@pytest.fixture
def ascending_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand out ids that ascend with the texts.

    Real ids are random, which makes the defect flaky rather than
    reproducible. `list()` ends `ORDER BY m.recorded_at DESC, m.id DESC`, so
    ascending ids make it return the texts in reverse alphabetical order --
    the exact opposite of the stated tiebreaker. A run without that
    tiebreaker then fails every time instead of one time in twenty-eight.
    """
    counter = itertools.count(1)
    monkeypatch.setattr(
        store_module, "_new_id", lambda: f"mem_20261115000000_{next(counter):04d}"
    )


def _kept_lines(path, facts=_FACTS) -> list[str]:
    with MemoryVault(path) as vault:
        for text in facts:
            vault.remember(text, kind="profile", origin="imported")
        return [
            line[2:]
            for line in vault.profile_block().splitlines()
            if line.startswith("- ")
        ]


def test_the_block_keeps_the_same_lines_whatever_the_ids_are(
    tmp_path, ascending_ids
) -> None:
    """The cap must cut by rank and a stated tiebreaker, never by id."""
    kept = _kept_lines(tmp_path / "vault.db")
    cap = store_module._PROFILE_KIND_CAPS["profile"]
    assert len(kept) == cap
    assert kept == sorted(_FACTS)[:cap]


def test_two_vaults_with_the_same_facts_build_the_same_block(tmp_path) -> None:
    """Inserted in opposite orders, with real random ids, both agree.

    This is the property that matters in production: two installs holding the
    same facts must put the same lines in front of the model.
    """
    first = _kept_lines(tmp_path / "a.db")
    second = _kept_lines(tmp_path / "b.db", facts=tuple(reversed(_FACTS)))
    assert first == second
