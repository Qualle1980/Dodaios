"""wiki-recall must return prose for filename-only / frontmatter-only hits.

Live 2026-11-14 case B: query matched a note whose term lived only outside the
body. ``hit.snippet`` was empty, recall dropped it silently, and the model
received title+path with no content — then said it found nothing while holding
the right note. ``SearchHit.preview`` already carries leading prose for this
case; the injector uses it, recall did not.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jarvis.memory.wiki.fts_index import ensure_schema, index_vault
from jarvis.memory.wiki.search import VaultSearch
from jarvis.plugins.tool.wiki_recall import WikiRecallTool

# Term only in the filename stem (indexed via frontmatter when stem != H1).
# Body deliberately avoids the query token.
_FILENAME_ONLY_PAGE = """---
type: lesson
slug: loop-lesson
---

# Loop lesson from November

The voice model called the same tool hundreds of times per turn.
A read cache and a withhold after the cap stopped the storm.
"""


def _vault(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    root = tmp_path / "vault"
    root.mkdir()
    # Stem carries the query token; H1 does not — that is how the indexer
    # puts the stem into the searchable frontmatter column.
    (root / "werkzeugschleife-november-lesson.md").write_text(
        _FILENAME_ONLY_PAGE, encoding="utf-8"
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    ensure_schema(conn)
    index_vault(root, conn)
    return root, conn


@pytest.mark.asyncio
async def test_filename_only_hit_returns_body_prose(tmp_path: Path) -> None:
    root, conn = _vault(tmp_path)
    search = VaultSearch(root, conn=conn)
    hits = search.search("werkzeugschleife", k=5)
    assert hits, "fixture must match on the filename stem"
    assert hits[0].snippet == "", "filename/frontmatter hit has empty body snippet"
    assert hits[0].preview, "search must still expose leading prose"

    # Length without the fallback (the live defect shape).
    bare = (
        f'## Wiki hits for "werkzeugschleife"\n'
        f"- **{hits[0].title}** [{hits[0].path.name}]"
    )
    before_chars = len(bare)

    tool = WikiRecallTool(search)
    result = await tool.execute({"query": "werkzeugschleife"}, ctx=None)
    assert result.success is True
    output = result.output or ""
    after_chars = len(output)

    # Must carry prose from the note, not only title and path.
    assert "voice model" in output or "read cache" in output or "withhold" in output
    assert "[no excerpt available" not in output
    # No output line that is only title + path (the silent-empty shape).
    for line in output.splitlines():
        if line.startswith("- **"):
            assert " — " in line, f"hit line has no content part: {line!r}"
            title_path_only = line.split(" — ", 1)[1].strip()
            assert not title_path_only.startswith("["), (
                f"hit line is title/path only: {line!r}"
            )

    assert after_chars > before_chars, (
        f"fallback must grow the payload: before={before_chars} after={after_chars}"
    )
    # Recorded for the report: same fixture call, before/after lengths.
    print(f"wiki-recall filename-hit bytes before={before_chars} after={after_chars}")


@pytest.mark.asyncio
async def test_empty_snippet_and_preview_is_loud(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A hit with no prose must not collapse to an empty string."""
    from jarvis.memory.wiki.search import SearchHit

    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("# Note\n\nbody\n", encoding="utf-8")

    class _EmptySearch:
        _root = root

        def search(self, query: str, *, k: int = 5):
            return [
                SearchHit(
                    title="EmptyHit",
                    path=root / "empty.md",
                    snippet="",
                    score=1.0,
                    preview="",
                )
            ]

    tool = WikiRecallTool(_EmptySearch())  # type: ignore[arg-type]
    with caplog.at_level("WARNING"):
        result = await tool.execute({"query": "anything"}, ctx=None)
    assert result.success is True
    assert "no excerpt available" in (result.output or "")
    assert any("empty snippet and preview" in r.getMessage() for r in caplog.records)
