"""The boot warm-up covers every leg a search pays for, not just the vault.

Measured 2026-11-07 against the owner's real vault and its six external
sources: a cold first search cost 724.5 ms against a 150 ms budget, of which
430.8 ms was per-source index building. The vault leg alone cost 3.3 ms. So
the whole overrun lived in the external legs -- exactly what ``warm_up`` was
introduced to prevent, for the one leg it covered.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from jarvis.memory.wiki.external_sources import ExternalSource, index_db_path
from jarvis.memory.wiki.fts_index import index_vault
from jarvis.memory.wiki.search import VaultSearch


def _vault(root: Path, name: str, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(f"# {name}\n\n{body}\n", encoding="utf-8")
    return root


def _canonical(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    root = _vault(tmp_path / "vault", "home", "the canonical page about pears")
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    index_vault(root, conn)
    return root, conn


def _source(tmp_path: Path, name: str) -> ExternalSource:
    root = _vault(tmp_path / name, "note", f"an external page about pears in {name}")
    return ExternalSource(name=name, path=root)


def test_warm_up_builds_every_external_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root, conn = _canonical(tmp_path)
    sources = (_source(tmp_path, "alpha"), _source(tmp_path, "beta"))
    search = VaultSearch(root, conn=conn, external_sources=sources)

    search.warm_up()

    for source in sources:
        assert index_db_path(source).exists(), f"{source.name} was not warmed"
        warmed = sqlite3.connect(str(index_db_path(source)))
        try:
            rows = warmed.execute("SELECT count(*) FROM wiki_fts").fetchone()[0]
        finally:
            warmed.close()
        assert rows == 1, f"{source.name} index is empty after warm_up"
    search.close()


def test_a_warmed_search_does_not_rebuild(tmp_path: Path, monkeypatch) -> None:
    """The point of warming: the turn must not pay the build a second time."""
    monkeypatch.chdir(tmp_path)
    root, conn = _canonical(tmp_path)
    source = _source(tmp_path, "alpha")
    search = VaultSearch(root, conn=conn, external_sources=(source,))
    search.warm_up()

    rebuilds: list[Path] = []
    import jarvis.memory.wiki.fts_index as fts

    real = fts.rebuild_index

    def spy(path, connection):  # noqa: ANN001, ANN202
        rebuilds.append(Path(path))
        return real(path, connection)

    monkeypatch.setattr(fts, "rebuild_index", spy)
    hits = search.search("pears", k=5)

    assert rebuilds == [], f"search rebuilt {rebuilds} although warm_up ran"
    assert {hit.source for hit in hits} == {"dodaios", "alpha"}
    search.close()
