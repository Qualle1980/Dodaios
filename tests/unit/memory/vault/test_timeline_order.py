"""A timeline comes out oldest first, even when the clock cannot tell.

`recorded_at` has one-second resolution and a memory id ends in random hex, so
two versions written in the same second could be ordered by neither. `about()`
sorted by exactly those two, which made "where did I live before?" answer in
arbitrary order - measured: Berlin before Hamburg in 2 of 6 runs, and the
single-run test that covered it passed either way.

The supersede links are what actually order two versions, which is why
`history()` already followed them. Now `about()` does too.
"""
from __future__ import annotations

import pytest

from jarvis.memory.vault import MemoryVault

#: Enough runs that a coin-flip ordering cannot pass by luck: the bug showed in
#: a third of runs, so twelve all-correct runs is a ~1 in 10^2 fluke at worst.
RUNS = 12


@pytest.fixture
def vault(tmp_path):
    with MemoryVault(tmp_path / "vault.db") as handle:
        yield handle


def _moved(vault: MemoryVault) -> None:
    first = vault.remember("Ahmad lives in Hamburg.", "profile", entities=["ahmad"])
    assert first.memory is not None
    second = vault.supersede(first.memory.id, "Ahmad lives in Berlin.", origin="user")
    assert second.memory is not None


@pytest.mark.parametrize("run", range(RUNS))
def test_about_puts_the_older_fact_first_every_time(tmp_path, run: int) -> None:
    with MemoryVault(tmp_path / f"vault-{run}.db") as vault:
        _moved(vault)

        texts = [m.text for m in vault.about("ahmad")]

    assert texts == ["Ahmad lives in Hamburg.", "Ahmad lives in Berlin."]


@pytest.mark.parametrize("run", range(RUNS))
def test_history_puts_the_older_fact_first_every_time(tmp_path, run: int) -> None:
    with MemoryVault(tmp_path / f"vault-{run}.db") as vault:
        _moved(vault)
        latest = next(m for m in vault.list() if "Berlin" in m.text)

        texts = [m.text for m in vault.history(latest.id)]

    assert texts == ["Ahmad lives in Hamburg.", "Ahmad lives in Berlin."]


def test_a_three_step_chain_reads_forwards(vault) -> None:
    first = vault.remember("Ahmad lives in Kiel.", "profile", entities=["ahmad"])
    second = vault.supersede(first.memory.id, "Ahmad lives in Hamburg.", origin="user")
    third = vault.supersede(second.memory.id, "Ahmad lives in Berlin.", origin="user")

    assert [m.text for m in vault.about("ahmad")] == [
        "Ahmad lives in Kiel.",
        "Ahmad lives in Hamburg.",
        "Ahmad lives in Berlin.",
    ]
    assert [m.text for m in vault.history(third.memory.id)] == [
        "Ahmad lives in Kiel.",
        "Ahmad lives in Hamburg.",
        "Ahmad lives in Berlin.",
    ]


def test_memories_from_different_chains_stay_in_time_order(tmp_path) -> None:
    """Across chains the clock leads; depth only breaks a tie within a second."""

    class Clock:
        """A settable clock. One write reads it more than once (the audit row)."""

        value = "2026-01-01T00:00:00+00:00"

        def __call__(self) -> str:
            return self.value

    clock = Clock()
    with MemoryVault(tmp_path / "vault.db", clock=clock) as vault:
        vault.remember("Ahmad drives a van.", "profile", entities=["ahmad"])
        clock.value = "2026-02-01T00:00:00+00:00"
        vault.remember("Ahmad plays golf.", "profile", entities=["ahmad"])
        clock.value = "2026-03-01T00:00:00+00:00"
        vault.remember("Ahmad lives in Berlin.", "profile", entities=["ahmad"])

        texts = [m.text for m in vault.about("ahmad")]

    assert texts == [
        "Ahmad drives a van.",
        "Ahmad plays golf.",
        "Ahmad lives in Berlin.",
    ]



def test_forgetting_one_version_erases_the_whole_chain(vault) -> None:
    """Not an ordering case - the guarantee that made one of my tests wrong.

    `forget` walks the supersede links in BOTH directions, so forgetting where
    someone used to live also erases where they live now. That is the privacy
    promise, not a bug: a half-erased chain would still say the fact existed.
    """
    first = vault.remember("Ahmad lives in Hamburg.", "profile", entities=["ahmad"])
    second = vault.supersede(first.memory.id, "Ahmad lives in Berlin.", origin="user")

    erased = vault.forget(first.memory.id)

    assert erased == 2
    assert vault.about("ahmad") == []
    assert vault.get(second.memory.id) is None
    assert len(vault.tombstones()) == 2
