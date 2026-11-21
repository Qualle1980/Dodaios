"""Tests for .agent-context/tools/context.py (the lesson FTS5 index)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / ".agent-context" / "tools" / "context.py"


def _load():
    spec = importlib.util.spec_from_file_location("agent_context_tool", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ctx = _load()


def _has_fts5() -> bool:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


pytestmark = pytest.mark.skipif(not _has_fts5(), reason="sqlite3 built without FTS5")


def _lesson(folder: Path, name: str, status: str, title: str, body: str, tags: str = "") -> Path:
    path = folder / name
    path.write_text(
        f"---\ntitle: {title}\nstatus: {status}\ntags: [{tags}]\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def store(tmp_path: Path) -> tuple[Path, Path]:
    lessons = tmp_path / "lessons"
    lessons.mkdir()
    _lesson(
        lessons,
        "worktree.md",
        "verified",
        "Orphaned worktree lock",
        "A zero-byte index.lock left by a killed git process.",
        "git, worktree",
    )
    _lesson(
        lessons,
        "guess.md",
        "draft",
        "Unproven worktree theory",
        "Maybe the worktree lock comes from antivirus.",
    )
    _lesson(lessons, "_TEMPLATE.md", "verified", "Template", "worktree placeholder")
    return lessons, tmp_path / "cache" / "lessons.sqlite"


def test_search_returns_only_verified_by_default(store):
    lessons, db = store
    hits = ctx.search("worktree lock", lessons_dir=lessons, db=db)
    assert [h["path"] for h in hits] == ["worktree.md"]


def test_all_flag_includes_drafts_and_skips_templates(store):
    lessons, db = store
    paths = {
        h["path"] for h in ctx.search("worktree", include_all=True, lessons_dir=lessons, db=db)
    }
    assert paths == {"worktree.md", "guess.md"}


def test_tags_are_searchable(store):
    lessons, db = store
    assert [h["path"] for h in ctx.search("git", lessons_dir=lessons, db=db)] == ["worktree.md"]


def test_cache_follows_edits_without_manual_rebuild(store):
    lessons, db = store
    assert ctx.search("antivirus", lessons_dir=lessons, db=db) == []
    _lesson(
        lessons,
        "guess.md",
        "verified",
        "Antivirus lock",
        "Proven: antivirus held the file open, with a longer body now.",
    )
    assert [h["path"] for h in ctx.search("antivirus", lessons_dir=lessons, db=db)] == ["guess.md"]
    (lessons / "guess.md").unlink()
    assert ctx.search("antivirus", lessons_dir=lessons, db=db) == []


def test_deleted_or_corrupt_cache_is_rebuilt(store):
    lessons, db = store
    assert ctx.rebuild(lessons, db) == 2
    db.write_bytes(b"not a database")
    assert [h["path"] for h in ctx.search("index.lock", lessons_dir=lessons, db=db)] == [
        "worktree.md"
    ]
    db.unlink()
    assert ctx.search("zero byte", lessons_dir=lessons, db=db)


@pytest.mark.parametrize("query", ['"', "AND OR NOT", "lock*)(", "NEAR(", "-git"])
def test_fts_syntax_in_query_never_raises(store, query):
    lessons, db = store
    ctx.search(query, lessons_dir=lessons, db=db)


def test_empty_query_returns_nothing(store):
    lessons, db = store
    assert ctx.search("  ...  ", lessons_dir=lessons, db=db) == []


def test_lesson_without_frontmatter_is_a_draft(tmp_path: Path):
    path = tmp_path / "plain.md"
    path.write_text("# Plain heading\n\nbody", encoding="utf-8")
    lesson = ctx.parse_lesson(path)
    assert (lesson.title, lesson.status) == ("Plain heading", "draft")


def test_cli_search_and_rebuild(store, capsys):
    lessons, db = store
    base = ["--lessons", str(lessons), "--db", str(db)]
    assert ctx.main([*base, "rebuild"]) == 0
    assert "indexed 2 lessons" in capsys.readouterr().out
    assert ctx.main([*base, "search", "worktree", "--json"]) == 0
    assert '"worktree.md"' in capsys.readouterr().out
    assert ctx.main([*base, "search", "nothing-matches-this"]) == 0
    assert "no matching lessons" in capsys.readouterr().out


def test_shipped_lessons_parse():
    for path in ctx.lesson_files(ctx.DEFAULT_LESSONS):
        assert ctx.parse_lesson(path).status in {"verified", "draft", "superseded"}, path.name
