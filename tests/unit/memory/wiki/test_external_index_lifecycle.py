"""External-source index lifecycle: built and refreshed off the turn.

A search reads only the last committed on-disk index; building and
reconciling happen in warm-up or a background refresh, one at a time.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import jarvis.memory.wiki.search as search_mod
from jarvis.brain.wiki_context import WikiContextInjector
from jarvis.memory.wiki import fts_index
from jarvis.memory.wiki.external_sources import ExternalSource
from jarvis.memory.wiki.search import VaultSearch, refresh_external_index


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        search_mod, "index_db_path", lambda s: tmp_path / "idx" / f"{s.name}.db"
    )
    monkeypatch.setattr(search_mod, "_external_refreshed_at", {})


def _source(tmp_path: Path, pages: int = 3) -> ExternalSource:
    root = tmp_path / "obsidian"
    root.mkdir()
    for i in range(pages):
        (root / f"note{i}.md").write_text(f"# Note {i}\n\nFiller text {i}.\n", "utf-8")
    (root / "jarvis.md").write_text("# Jarvis\n\nProvenance telemetry landed.\n", "utf-8")
    return ExternalSource(name="obsidian", path=root.resolve())


def _vault(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    vault = tmp_path / "vault"
    vault.mkdir()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    fts_index.ensure_schema(conn)
    return vault, conn


def _wait_refresh(name: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while name not in search_mod._external_refreshed_at and time.monotonic() < deadline:
        time.sleep(0.01)
    lock = search_mod._external_lock(name)
    lock.acquire(timeout=timeout)
    lock.release()


def test_cold_search_does_not_build_and_reports_not_ready(tmp_path: Path) -> None:
    source = _source(tmp_path, pages=400)
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    t0 = time.perf_counter()
    hits = search.search("Provenance")
    cold_ms = (time.perf_counter() - t0) * 1000
    assert hits == []
    assert search.index_ready() is False
    # The background refresh builds it; the next search finds the note.
    _wait_refresh("obsidian")
    hits = search.search("Provenance")
    assert [h.title for h in hits] == ["Jarvis"]
    assert hits[0].source == "obsidian"
    assert search.index_ready() is True
    assert cold_ms < 150, f"cold search took {cold_ms:.1f} ms"


def test_warm_search_is_fast_after_warm_up(tmp_path: Path) -> None:
    source = _source(tmp_path, pages=400)
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    search.warm_up()
    t0 = time.perf_counter()
    hits = search.search("Provenance telemetry")
    warm_ms = (time.perf_counter() - t0) * 1000
    assert hits and hits[0].title == "Jarvis"
    assert warm_ms < 50, f"warm search took {warm_ms:.1f} ms"


def test_changed_and_deleted_files_are_reconciled(tmp_path: Path) -> None:
    source = _source(tmp_path)
    assert refresh_external_index(source)
    note = source.path / "note0.md"
    note.write_text("# Note 0\n\nNow mentions Zebrafink.\n", "utf-8")
    later = time.time() + 5
    os.utime(note, (later, later))
    (source.path / "note1.md").unlink()
    assert refresh_external_index(source)
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    search_mod._external_refreshed_at[source.name] = time.monotonic()
    assert [h.title for h in search.search("Zebrafink")] == ["Note 0"]
    assert search.search("Filler 1") == [] or all(
        h.path.name != "note1.md" for h in search.search("Filler 1")
    )


def test_failed_refresh_keeps_the_last_valid_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    assert refresh_external_index(source)
    (source.path / "jarvis.md").write_text("# Jarvis\n\nReplaced.\n", "utf-8")
    later = time.time() + 5
    os.utime(source.path / "jarvis.md", (later, later))

    def _boom(*_a, **_k):
        raise OSError("disk hiccup")

    monkeypatch.setattr(fts_index, "_upsert_one", _boom)
    assert refresh_external_index(source) is False
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    search_mod._external_refreshed_at[source.name] = time.monotonic()
    assert [h.title for h in search.search("Provenance")] == ["Jarvis"]


def test_concurrent_search_and_refresh_do_not_conflict(tmp_path: Path) -> None:
    source = _source(tmp_path, pages=300)
    assert refresh_external_index(source)
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    search_mod._external_refreshed_at[source.name] = time.monotonic()
    errors: list[BaseException] = []
    stop = threading.Event()

    def _writer() -> None:
        try:
            for i in range(15):
                p = source.path / f"note{i}.md"
                p.write_text(f"# Note {i}\n\nEdited {i}.\n", "utf-8")
                t = time.time() + 10 + i
                os.utime(p, (t, t))
                refresh_external_index(source)
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assert
            errors.append(exc)
        finally:
            stop.set()

    writer = threading.Thread(target=_writer)
    writer.start()
    found = 0
    while not stop.is_set():
        found += len(search.search("Provenance"))
    writer.join()
    assert errors == []
    assert found > 0
    # Only one refresh per source may run: a second caller returns at once.
    lock = search_mod._external_lock(source.name)
    with lock:
        assert refresh_external_index(source) is False


async def test_injector_reports_index_not_ready_and_leaves_no_rebuild(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(tmp_path, pages=50)
    vault, conn = _vault(tmp_path)
    search = VaultSearch(vault, conn=conn, external_sources=(source,))
    injector = WikiContextInjector(search=search, relevance_gate=False)
    caplog.set_level("INFO")
    out = await injector.maybe_inject(
        user_text="Provenance telemetry status", system_prompt="BASE"
    )
    assert out == "BASE"
    assert "reason=index_not_ready" in caplog.text
    # The only thing left running is the single bounded background refresh.
    _wait_refresh("obsidian")
    assert not [t for t in threading.enumerate() if t.name.startswith("wiki-external-index")]
    await asyncio.sleep(0)
