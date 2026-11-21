#!/usr/bin/env python3
"""Search the agent lessons in ``.agent-context/lessons`` through SQLite FTS5.

The Markdown files are the source of truth. The SQLite file under
``.agent-context/.cache/`` is only a cache: ``search`` rebuilds it whenever a
lesson was added, removed or changed, and deleting it loses nothing.

Usage::

    python .agent-context/tools/context.py rebuild
    python .agent-context/tools/context.py search "worktree lock" [--all] [--limit N] [--json]

By default only lessons with ``status: verified`` are returned; ``--all``
includes drafts and superseded notes. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LESSONS = ROOT / "lessons"
DEFAULT_DB = ROOT / ".cache" / "lessons.sqlite"
SCHEMA_VERSION = "1"
_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Lesson:
    path: str
    title: str
    status: str
    tags: str
    body: str


def parse_lesson(path: Path) -> Lesson:
    """Read one lesson; a missing frontmatter yields status ``draft``."""
    text = path.read_text(encoding="utf-8-sig")
    meta: dict[str, str] = {}
    body = text
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for end, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                for raw in lines[1:end]:
                    key, sep, value = raw.partition(":")
                    if sep:
                        meta[key.strip().lower()] = value.strip().strip("\"'")
                body = "\n".join(lines[end + 1 :])
                break
    title = meta.get("title", "")
    if not title:
        heading = next((ln for ln in body.splitlines() if ln.startswith("# ")), "")
        title = heading[2:].strip() or path.stem
    tags = meta.get("tags", "").strip("[]")
    tags = " ".join(t.strip() for t in tags.split(",") if t.strip())
    return Lesson(
        path=path.name,
        title=title,
        status=meta.get("status", "draft").lower(),
        tags=tags,
        body=body.strip(),
    )


def lesson_files(lessons_dir: Path) -> list[Path]:
    """Every lesson file; names starting with ``_`` are templates, not lessons."""
    if not lessons_dir.is_dir():
        return []
    return sorted(p for p in lessons_dir.glob("*.md") if not p.name.startswith("_"))


def fingerprint(lessons_dir: Path) -> str:
    """Name, size and mtime of every lesson: any edit changes it."""
    parts = [SCHEMA_VERSION]
    for p in lesson_files(lessons_dir):
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def rebuild(lessons_dir: Path = DEFAULT_LESSONS, db: Path = DEFAULT_DB) -> int:
    """Recreate the index from the Markdown files and return the lesson count."""
    lessons = [parse_lesson(p) for p in lesson_files(lessons_dir)]
    db.parent.mkdir(parents=True, exist_ok=True)
    # The file is only a cache: start from nothing, which also replaces a
    # corrupt or foreign file that sqlite would refuse to open.
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    try:
        with con:
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            con.execute(
                "CREATE VIRTUAL TABLE lessons USING fts5("
                "title, tags, body, path UNINDEXED, status UNINDEXED, "
                "tokenize = 'unicode61 remove_diacritics 2')"
            )
            con.executemany(
                "INSERT INTO lessons (title, tags, body, path, status) VALUES (?, ?, ?, ?, ?)",
                [(x.title, x.tags, x.body, x.path, x.status) for x in lessons],
            )
            con.execute(
                "INSERT INTO meta (key, value) VALUES ('fingerprint', ?)",
                (fingerprint(lessons_dir),),
            )
    finally:
        con.close()
    return len(lessons)


def _is_fresh(lessons_dir: Path, db: Path) -> bool:
    if not db.is_file():
        return False
    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
    except sqlite3.DatabaseError:
        # A corrupt or foreign cache file is simply stale: the rebuild replaces it.
        return False
    finally:
        con.close()
    return bool(row) and row[0] == fingerprint(lessons_dir)


def to_match_query(query: str) -> str:
    """Quote every word so FTS5 operators in user text cannot raise a syntax error."""
    return " ".join(f'"{tok}"' for tok in _TOKEN.findall(query))


def search(
    query: str,
    *,
    include_all: bool = False,
    limit: int = 5,
    lessons_dir: Path = DEFAULT_LESSONS,
    db: Path = DEFAULT_DB,
) -> list[dict[str, str]]:
    """Return the best matching lessons; rebuilds a stale cache first."""
    match = to_match_query(query)
    if not match:
        return []
    if not _is_fresh(lessons_dir, db):
        rebuild(lessons_dir, db)
    sql = (
        "SELECT path, title, status, snippet(lessons, 2, '[', ']', ' ... ', 16) "
        "FROM lessons WHERE lessons MATCH ?"
    )
    params: list[object] = [match]
    if not include_all:
        sql += " AND status = 'verified'"
    sql += " ORDER BY bm25(lessons, 5.0, 3.0, 1.0) LIMIT ?"
    params.append(max(1, limit))
    con = sqlite3.connect(db)
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return [{"path": p, "title": t, "status": s, "snippet": snip} for p, t, s, snip in rows]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search .agent-context lessons (SQLite FTS5).")
    parser.add_argument("--lessons", type=Path, default=DEFAULT_LESSONS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rebuild", help="recreate the cache from lessons/*.md")
    s = sub.add_parser("search", help="full-text search over the lessons")
    s.add_argument("query")
    s.add_argument("--all", action="store_true", help="include non-verified lessons")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "rebuild":
        print(f"indexed {rebuild(args.lessons, args.db)} lessons -> {args.db}")
        return 0
    hits = search(
        args.query, include_all=args.all, limit=args.limit, lessons_dir=args.lessons, db=args.db
    )
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    elif not hits:
        print("no matching lessons")
    else:
        for h in hits:
            print(f"{h['path']} [{h['status']}] {h['title']}\n    {h['snippet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
