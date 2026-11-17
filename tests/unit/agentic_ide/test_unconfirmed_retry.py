"""An unconfirmed model list is not re-read on every access, and never picked for a size (2026-11-17)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from jarvis.agentic_ide import model_lists as ml
from jarvis.workspace.task_size import pick_for_size


def _store(tmp_path: Path, now: list[float]) -> ml.ModelListStore:
    return ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(
            grok_cache=tmp_path / "grok.json",
            codex_cache=tmp_path / "codex.json",
            claude_catalog_file=tmp_path / "claude.json",
            codex_lister=None,
        ),
        now=lambda: now[0],
        refresh_ttl_s=600.0,
    )


def test_unconfirmed_list_is_answered_from_memory_within_a_minute(tmp_path: Path) -> None:
    now = [1_000.0]
    lists = _store(tmp_path, now)
    calls: list[str] = []

    def unconfirmed(cli: str, *, read_at: float):  # same tuple shape as _read_live
        calls.append(cli)
        return (), read_at, "cache_unconfirmed", False, "probe failed"

    lists._read_live = unconfirmed  # type: ignore[method-assign]
    lists.ensure("codex")
    now[0] = 1_030.0
    lists.ensure("codex")  # the description path asks again within seconds
    assert len(calls) == 1

    now[0] = 1_061.0
    lists.ensure("codex")  # past the retry window: the CLI is asked once more
    assert len(calls) == 2


def test_an_unconfirmed_list_never_yields_a_size_pick(tmp_path: Path, live_store: ml.ModelListStore) -> None:
    class Unconfirmed:
        def ensure(self, cli: str) -> ml.CliModelList:
            return dataclasses.replace(live_store.ensure(cli), confirmed=False)

    pick = pick_for_size("claude", "simple", store=Unconfirmed())  # type: ignore[arg-type]
    assert not pick.model  # empty pick: the CLI default is used instead


@pytest.fixture
def live_store(tmp_path: Path) -> ml.ModelListStore:
    """A store whose claude list is confirmed, so the baseline pick exists."""
    from tests.unit.workspace.test_task_size import _write_caches  # reuse fixture data

    caches = _write_caches(tmp_path)
    return ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=None,
        ),
        now=lambda: 2_000.0,
        refresh_ttl_s=600.0,
    )
